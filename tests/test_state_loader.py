# tests/test_state_loader.py
"""
Behavioral spec for kinetix/utils/state_loader.py (LAMMPS dump parsing and
kMC-state injection).

Real artifacts, no hardcoded config literals:
  - data/parameters/defects/PZT_ZrPbO3_defects_config.yaml (DefectsConfig.from_yaml)
  - data/parameters/defects/VCM_HfO2_defects_config.yaml   (DefectsConfig.from_yaml)
  - the species-id maps are produced by the REAL production
    KMCSimulator._species_id_gen (on an uninitialized instance), so dump
    type ids used here always follow the shipped configs.

Synthetic LAMMPS dumps are written into tmp_path by ``write_dump``.

BEHAVIOR NOTES (all pinned by the tests below):
  * Box bounds are consumed but not returned: parse_lammps_dump yields only
    {'timestep', 'atoms'} (the docstring's "box bounds" claim is stale). This is
    intentional - the simulation domain is predefined by the grid.
  * There is no periodic-image folding: coordinates are matched against the raw
    grid positions, so a dump atom written past the upper face is dropped (with
    a distance warning) instead of being wrapped. Dumps are expected to be
    written pre-wrapped.
  * A multi-frame dump is parsed as a single (last) frame: 'atoms' restarts at
    every ITEM: ATOMS block and 'timestep' holds the last frame's value.
  * Malformed rows raise ValueError carrying the offending line number.
  * load_state_from_dump reports through the module logger and defaults a
    missing charge column to 0 (neutral).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kinetix.configs.defect_config import DefectsConfig
from kinetix.lattice.simulator import KMCSimulator
from kinetix.lattice.defect import make_empty_defect
from kinetix.utils.state_loader import (
    decode_species_key_passivation,
    load_state_from_dump,
    parse_lammps_dump,
)

PARAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "parameters"

DEFAULT_COLUMNS = ("id", "type", "x", "y", "z", "charge")
BOX = (0.0, 10.0)


# =============================================================================
# Real-config fixtures
# =============================================================================

def _load_defects_dict(filename: str) -> dict:
    """Load a shipped defects YAML through the production loader."""
    return DefectsConfig.from_yaml(PARAMS_DIR / "defects" / filename).to_dict()


@pytest.fixture(scope="module")
def pzt_defects_dict() -> dict:
    """REAL PZT defect config (oxygen_vacancy declares max_passivation_level)."""
    return _load_defects_dict("PZT_ZrPbO3_defects_config.yaml")


@pytest.fixture(scope="module")
def vcm_defects_dict() -> dict:
    """REAL VCM defect config (no defect declares a passivation level)."""
    return _load_defects_dict("VCM_HfO2_defects_config.yaml")


def build_species_maps(defects_config: dict) -> tuple[dict, dict]:
    """Build (SPECIES_TYPE_MAP, SPECIES_ID_TO_TYPE) with the REAL production
    KMCSimulator._species_id_gen so ids follow production ordering."""
    lattice = KMCSimulator.__new__(KMCSimulator)  # skip __init__/physics
    lattice.defects_config = defects_config
    KMCSimulator._species_id_gen(lattice)
    return lattice.SPECIES_TYPE_MAP, lattice.SPECIES_ID_TO_TYPE


@pytest.fixture(scope="module")
def pzt_species_maps(pzt_defects_dict) -> tuple[dict, dict]:
    return build_species_maps(pzt_defects_dict)


# =============================================================================
# Dump writer + mock system
# =============================================================================

def write_dump(path, frames, columns=DEFAULT_COLUMNS, box=BOX):
    """Write a synthetic LAMMPS dump.

    frames: iterable of (timestep, [atom_dict]) - one dict per atom keyed by
    the column names. Returns the path for convenience.
    """
    lines = []
    for timestep, atoms in frames:
        lines.append("ITEM: TIMESTEP")
        lines.append(str(timestep))
        lines.append("ITEM: NUMBER OF ATOMS")
        lines.append(str(len(atoms)))
        lines.append("ITEM: BOX BOUNDS pp pp pp")
        for _ in range(3):
            lines.append(f"{box[0]} {box[1]}")
        lines.append("ITEM: ATOMS " + " ".join(columns))
        for entry in atoms:
            lines.append(" ".join(str(entry[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n")
    return path


def atom(atom_id, type_id, x, y, z, charge=0.0):
    return {"id": atom_id, "type": type_id, "x": x, "y": y, "z": z, "charge": charge}


class MockSite:
    """Minimal Site stand-in (position / species / charge / passivation)."""

    def __init__(self, position, specie="O", ion_charge=0):
        self.position = position
        # State lives on the Defect (Phase 6 removed Site's flat aliases).
        self.defect = make_empty_defect()
        self.defect.chemical_specie = specie
        self.defect.charge = ion_charge
        self.defect.passivation_level = 0


def make_site(position, specie="O", ion_charge=0):
    return MockSite(position, specie, ion_charge)


def make_grid():
    """3-site grid: (0,0,0), (2,0,0), (0,2,0) - 2 Angstrom spacing."""
    return {
        (0, 0, 0): make_site((0.0, 0.0, 0.0)),
        (1, 0, 0): make_site((2.0, 0.0, 0.0)),
        (0, 1, 0): make_site((0.0, 2.0, 0.0)),
    }


class MockSystemState:
    """Duck-typed KMCSimulator stand-in for load_state_from_dump.

    Records every injected species and topology rebuild so the tests can assert
    the mapping/selection behavior without a real lattice.
    """

    def __init__(self, grid, defects_config, species_id_to_type=None):
        self.grid_crystal = grid
        self.defects_config = defects_config
        self.introduced = []
        self.topology_calls = []
        self.species_id_gen_calls = 0
        self.time = -1.0
        self.list_time = []
        if species_id_to_type is not None:
            self.SPECIES_ID_TO_TYPE = dict(species_id_to_type)

    @property
    def event_handler(self):
        """The load path calls ``system.event_handler.<name>`` after the global
        delegate cleanup; this recorder already implements those two methods, so
        it doubles as its own EventHandler."""
        return self

    def _species_id_gen(self):
        self.species_id_gen_calls += 1
        self.SPECIES_TYPE_MAP, self.SPECIES_ID_TO_TYPE = build_species_maps(
            self.defects_config
        )

    def _introduce_specie_site(self, idx, support_update_sites,
                               event_update_sites, chemical_specie, ion_charge):
        self.introduced.append(
            {"idx": idx, "specie": chemical_specie, "charge": ion_charge}
        )
        site = self.grid_crystal[idx]
        site.defect.chemical_specie = chemical_specie
        site.defect.charge = ion_charge
        event_update_sites.add(idx)

    def update_sites_topology(self, support_update_sites, event_update_sites):
        self.topology_calls.append(
            (set(support_update_sites), set(event_update_sites))
        )


# =============================================================================
# parse_lammps_dump
# =============================================================================

class TestParseLammpsDump:
    """Timestep, atom records, typing, and error surfaces."""

    def test_parses_timestep_and_atom_records(self, tmp_path):
        dump = write_dump(
            tmp_path / "traj.dump",
            [(100, [atom(1, 2, 0.5, 0.0, 0.0, -2.0),
                    atom(2, 3, 2.0, 0.0, 0.0, 1.0)])],
        )
        data = parse_lammps_dump(str(dump))

        assert data["timestep"] == pytest.approx(100.0)
        assert len(data["atoms"]) == 2
        first = data["atoms"][0]
        assert first["id"] == 1 and isinstance(first["id"], int)
        assert first["type"] == 2 and isinstance(first["type"], int)
        assert (first["x"], first["y"], first["z"]) == (0.5, 0.0, 0.0)
        assert first["charge"] == pytest.approx(-2.0)
        assert all(isinstance(first[k], float) for k in ("x", "y", "z", "charge"))
        assert set(first) == set(DEFAULT_COLUMNS)

    def test_atom_columns_are_read_by_header_name(self, tmp_path):
        # Columns in a non-canonical order must still map onto the right fields.
        columns = ("type", "id", "z", "y", "x", "charge")
        dump = write_dump(
            tmp_path / "reordered.dump",
            [(7, [atom(11, 4, 1.0, 2.0, 3.0, 0.5)])],
            columns=columns,
        )
        parsed = parse_lammps_dump(str(dump))["atoms"][0]
        assert parsed == {
            "id": 11, "type": 4, "x": 1.0, "y": 2.0, "z": 3.0, "charge": 0.5,
        }

    def test_blank_lines_inside_atom_block_are_skipped(self, tmp_path):
        path = tmp_path / "blank.dump"
        write_dump(path, [(5, [atom(1, 2, 0.0, 0.0, 0.0, 0.0)])])
        text = path.read_text()
        path.write_text(text.replace("1 2 0.0", "\n1 2 0.0"))
        data = parse_lammps_dump(str(path))
        assert len(data["atoms"]) == 1
        assert data["atoms"][0]["id"] == 1

    def test_box_bounds_values_are_consumed_but_not_returned(self, tmp_path):
        # Documented design: the domain comes from the grid, not from the dump,
        # so parse_lammps_dump exposes only timestep + atoms.
        dump = write_dump(
            tmp_path / "box.dump",
            [(0, [atom(1, 2, 0.0, 0.0, 0.0, 0.0)])],
            box=(-5.0, 5.0),
        )
        data = parse_lammps_dump(str(dump))
        assert set(data) == {"timestep", "atoms"}
        assert data["atoms"][0]["x"] == pytest.approx(0.0)

    def test_empty_file_returns_defaults(self, tmp_path):
        empty = tmp_path / "empty.dump"
        empty.write_text("")
        data = parse_lammps_dump(str(empty))
        assert data == {"timestep": 0.0, "atoms": []}

    def test_headers_without_atom_rows(self, tmp_path):
        path = tmp_path / "noatoms.dump"
        write_dump(path, [(42, [])])
        data = parse_lammps_dump(str(path))
        assert data["timestep"] == pytest.approx(42.0)
        assert data["atoms"] == []

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_lammps_dump(str(tmp_path / "does_not_exist.dump"))

    def test_malformed_short_row_raises_value_error_with_line_number(self, tmp_path):
        path = tmp_path / "short.dump"
        write_dump(path, [(1, [atom(1, 2, 0.0, 0.0, 0.0, 0.0)])])
        lines = path.read_text().splitlines()
        lines[-1] = "1 2 0.0 0.0"  # drop two columns
        path.write_text("\n".join(lines) + "\n")

        with pytest.raises(ValueError) as excinfo:
            parse_lammps_dump(str(path))
        message = str(excinfo.value)
        assert "Malformed row at line" in message
        assert str(len(lines)) in message
        assert "expected 6" in message and "got 4" in message

    def test_multi_frame_dump_keeps_only_the_last_frame(self, tmp_path):
        dump = write_dump(
            tmp_path / "multi.dump",
            [
                (10, [atom(1, 2, 0.0, 0.0, 0.0, 0.0),
                      atom(2, 2, 1.0, 0.0, 0.0, 0.0)]),
                (20, [atom(3, 4, 5.0, 5.0, 5.0, 1.0)]),
            ],
        )
        data = parse_lammps_dump(str(dump))
        assert data["timestep"] == pytest.approx(20.0)
        assert [entry["id"] for entry in data["atoms"]] == [3]

    def test_non_numeric_value_raises(self, tmp_path):
        path = tmp_path / "badtype.dump"
        write_dump(path, [(1, [atom(1, 2, 0.0, 0.0, 0.0, 0.0)])])
        lines = path.read_text().splitlines()
        lines[-1] = lines[-1].replace("1 2 0.0", "1 O 0.0", 1)
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError):
            parse_lammps_dump(str(path))


# =============================================================================
# Species-id maps (produced by the real KMCSimulator._species_id_gen)
# =============================================================================

class TestSpeciesIdMaps:
    """The dump type ids follow the shipped defect configs."""

    def test_ids_are_unique_and_contiguous_from_one(self, pzt_species_maps):
        type_map, id_to_type = pzt_species_maps
        assert set(type_map.values()) == set(range(1, len(type_map) + 1))
        assert id_to_type == {v: k for k, v in type_map.items()}

    def test_passivation_levels_expand_into_separate_ids(self, pzt_species_maps,
                                                         pzt_defects_dict):
        type_map, _ = pzt_species_maps
        vacancy = pzt_defects_dict["oxygen_vacancy"]
        symbol = vacancy["symbol"]
        max_level = vacancy["max_passivation_level"]

        assert max_level >= 1, "fixture expects a passivated defect in the real config"
        for level in range(max_level + 1):
            assert f"{symbol}_{level}" in type_map
        # One level beyond the configured maximum must not exist.
        assert f"{symbol}_{max_level + 1}" not in type_map

    def test_unpassivated_defects_use_their_bare_symbol(self, pzt_species_maps,
                                                        pzt_defects_dict):
        type_map, _ = pzt_species_maps
        for name, cfg in pzt_defects_dict.items():
            if cfg.get("max_passivation_level", 0) == 0:
                assert cfg["symbol"] in type_map, f"{name} missing from species map"
                assert f"{cfg['symbol']}_0" not in type_map

    def test_vcm_config_produces_bare_symbols_only(self, vcm_defects_dict):
        type_map, _ = build_species_maps(vcm_defects_dict)
        assert not any(key.endswith("_0") for key in type_map), (
            "no VCM defect declares a passivation level"
        )
        for cfg in vcm_defects_dict.values():
            assert cfg["symbol"] in type_map

    def test_every_map_key_round_trips_through_the_decoder(self, pzt_species_maps,
                                                           pzt_defects_dict):
        type_map, _ = pzt_species_maps
        for species_key in type_map:
            chemical_specie, level, defect_name = decode_species_key_passivation(
                species_key, pzt_defects_dict
            )
            assert defect_name is not None, f"{species_key} did not decode to a defect"
            assert chemical_specie == pzt_defects_dict[defect_name]["symbol"]
            assert 0 <= level <= pzt_defects_dict[defect_name].get(
                "max_passivation_level", 0
            )


# =============================================================================
# decode_species_key_passivation
# =============================================================================

class TestDecodeSpeciesKeyPassivation:
    """'V_O_1' -> ('V_O', 1, 'oxygen_vacancy') with the real PZT config."""

    def test_bare_symbol_decodes_with_level_zero(self, pzt_defects_dict):
        assert decode_species_key_passivation("V_O", pzt_defects_dict) == (
            "V_O", 0, "oxygen_vacancy"
        )

    def test_passivated_symbol_decodes_symbol_and_level(self, pzt_defects_dict):
        max_level = pzt_defects_dict["oxygen_vacancy"]["max_passivation_level"]
        assert max_level >= 1
        assert decode_species_key_passivation("V_O_1", pzt_defects_dict) == (
            "V_O", 1, "oxygen_vacancy"
        )

    def test_multiple_defects_are_disambiguated_by_symbol(self, pzt_defects_dict):
        assert decode_species_key_passivation("H", pzt_defects_dict) == (
            "H", 0, "hydrogen_interstitial"
        )
        assert decode_species_key_passivation("H2", pzt_defects_dict) == (
            "H2", 0, "hydrogen_gas"
        )

    def test_level_beyond_max_is_unrecognized(self, pzt_defects_dict):
        max_level = pzt_defects_dict["oxygen_vacancy"]["max_passivation_level"]
        key = f"V_O_{max_level + 1}"
        assert decode_species_key_passivation(key, pzt_defects_dict) == (key, 0, None)

    def test_unknown_symbol_returns_no_defect(self, pzt_defects_dict):
        assert decode_species_key_passivation("Zr", pzt_defects_dict) == (
            "Zr", 0, None
        )

    def test_vcm_config_declares_no_passivation_levels(self, vcm_defects_dict):
        # 'V_O_1' is not a valid key when max_passivation_level is absent.
        assert decode_species_key_passivation("V_O_1", vcm_defects_dict) == (
            "V_O_1", 0, None
        )


# =============================================================================
# load_state_from_dump
# =============================================================================

def _type_id_for(pzt_defects_dict, species_key: str) -> int:
    """Type id that load_state_from_dump expects for a species key."""
    type_map, _ = build_species_maps(pzt_defects_dict)
    assert species_key in type_map, f"{species_key} not in the real species map"
    return type_map[species_key]


@pytest.fixture
def state_loader_logs():
    """Collect log records emitted by the state_loader module logger.

    pytest's ``caplog`` cannot be used here: ``kinetix.logging_config.setup_logging``
    (invoked by ``initialize_grid_crystal`` in several other test modules) sets
    ``propagate = False`` on the 'kinetix' logger, so records never reach the root
    handler that caplog attaches to. Attaching a collector directly to the module
    logger keeps this test independent of that global logging state.
    """
    logger = logging.getLogger("kinetix.utils.state_loader")
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collector()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


class TestLoadStateFromDump:
    """Mapping dump atoms onto grid sites and updating the system state."""

    def test_loads_defect_atom_onto_nearest_site(self, tmp_path, pzt_defects_dict):
        grid = make_grid()
        system = MockSystemState(grid, pzt_defects_dict)  # maps generated internally
        type_id = _type_id_for(pzt_defects_dict, "H")
        # Slightly off-site coordinate: the KDTree must still snap to (1,0,0).
        dump = write_dump(tmp_path / "state.dump",
                          [(1000, [atom(1, type_id, 2.05, 0.0, 0.0, 1.0)])])

        load_state_from_dump(system, str(dump))

        assert system.species_id_gen_calls == 1
        assert len(system.introduced) == 1
        entry = system.introduced[0]
        assert entry["idx"] == (1, 0, 0)
        assert entry["specie"] == "H"
        assert entry["charge"] == pytest.approx(1.0)
        assert grid[(1, 0, 0)].defect.chemical_specie == "H"
        # Time is taken from the dump timestep.
        assert system.time == pytest.approx(1000.0)
        assert system.list_time == pytest.approx([1000.0])

    def test_topology_rebuild_receives_the_updated_sites(self, tmp_path, pzt_defects_dict):
        grid = make_grid()
        system = MockSystemState(grid, pzt_defects_dict)
        type_id = _type_id_for(pzt_defects_dict, "H")
        dump = write_dump(tmp_path / "state.dump",
                          [(50, [atom(1, type_id, 0.0, 0.0, 0.0, 1.0)])])

        load_state_from_dump(system, str(dump))

        assert len(system.topology_calls) == 1
        support_sites, event_sites = system.topology_calls[0]
        assert event_sites == {(0, 0, 0)}
        assert support_sites == set()

    def test_existing_species_map_is_reused_without_regeneration(self, tmp_path,
                                                                 pzt_defects_dict):
        # Regression guard for the former NameError (undefined
        # 'system_SPECIES_ID_TO_TYPE'): a pre-existing map must be usable.
        _, id_to_type = build_species_maps(pzt_defects_dict)
        grid = make_grid()
        system = MockSystemState(grid, pzt_defects_dict,
                                 species_id_to_type=id_to_type)
        dump = write_dump(tmp_path / "state.dump",
                          [(1, [atom(1, _type_id_for(pzt_defects_dict, "H"),
                                     0.0, 0.0, 0.0, 1.0)])])

        load_state_from_dump(system, str(dump))  # must not raise NameError

        assert system.species_id_gen_calls == 0
        assert system.SPECIES_ID_TO_TYPE == id_to_type
        assert system.introduced[0]["specie"] == "H"

    def test_passivation_level_is_applied_to_the_site(self, tmp_path, pzt_defects_dict):
        grid = make_grid()
        system = MockSystemState(grid, pzt_defects_dict)
        type_id = _type_id_for(pzt_defects_dict, "V_O_1")
        dump = write_dump(tmp_path / "passivated.dump",
                          [(10, [atom(1, type_id, 0.0, 0.0, 0.0, 0.0)])])

        load_state_from_dump(system, str(dump))

        assert system.introduced[0]["specie"] == "V_O"
        assert grid[(0, 0, 0)].defect.passivation_level == 1

    def test_missing_charge_column_defaults_to_neutral(self, tmp_path, pzt_defects_dict):
        grid = make_grid()
        system = MockSystemState(grid, pzt_defects_dict)
        columns = ("id", "type", "x", "y", "z")
        type_id = _type_id_for(pzt_defects_dict, "H")
        dump = write_dump(
            tmp_path / "nocharge.dump",
            [(1, [{"id": 1, "type": type_id, "x": 0.0, "y": 0.0, "z": 0.0}])],
            columns=columns,
        )

        load_state_from_dump(system, str(dump))

        assert system.introduced[0]["charge"] == 0
        assert grid[(0, 0, 0)].defect.charge == 0

    def test_atom_far_from_every_site_is_skipped_with_warning(self, tmp_path,
                                                              pzt_defects_dict,
                                                              state_loader_logs):
        grid = make_grid()
        system = MockSystemState(grid, pzt_defects_dict)
        type_id = _type_id_for(pzt_defects_dict, "H")
        dump = write_dump(tmp_path / "offgrid.dump",
                          [(1, [atom(1, type_id, 5.0, 5.0, 5.0, 1.0)])])

        load_state_from_dump(system, str(dump))

        assert system.introduced == []
        assert any("No grid site found near" in rec.getMessage()
                   for rec in state_loader_logs)

    def test_coordinates_past_the_upper_face_are_not_wrapped(self, tmp_path,
                                                             pzt_defects_dict):
        # No periodic-image folding: an atom written just above the box top is
        # farther than the tolerance from every site and is therefore dropped.
        grid = make_grid()
        system = MockSystemState(grid, pzt_defects_dict)
        type_id = _type_id_for(pzt_defects_dict, "H")
        dump = write_dump(
            tmp_path / "unwrapped.dump",
            [(1, [atom(1, type_id, BOX[1] + 0.05, 0.0, 0.0, 1.0)])],
            box=BOX,
        )

        load_state_from_dump(system, str(dump), tolerance=0.1)

        assert system.introduced == []  # would be site (0,0,0) if wrapped

    def test_host_lattice_and_empty_types_are_skipped(self, tmp_path, pzt_defects_dict):
        grid = make_grid()
        # Inject a host-lattice entry (Zr) and an 'Empty' entry to exercise the
        # two skip paths; all other ids come from the real config.
        _, id_to_type = build_species_maps(pzt_defects_dict)
        id_to_type = dict(id_to_type)
        id_to_type[900] = "Zr"     # decodes to defect_name None -> host lattice
        id_to_type[901] = "Empty"  # explicit skip
        system = MockSystemState(grid, pzt_defects_dict, species_id_to_type=id_to_type)
        dump = write_dump(
            tmp_path / "mixed.dump",
            [(1, [atom(1, 900, 0.0, 0.0, 0.0, 0.0),
                  atom(2, 901, 2.0, 0.0, 0.0, 0.0),
                  atom(3, 999, 0.0, 2.0, 0.0, 0.0)])],  # 999 -> unknown id
        )

        load_state_from_dump(system, str(dump))

        assert system.introduced == []
        assert system.time == pytest.approx(1.0)

    def test_empty_dump_sets_time_and_rebuilds_with_empty_sets(self, tmp_path,
                                                               pzt_defects_dict):
        grid = make_grid()
        system = MockSystemState(grid, pzt_defects_dict)
        dump = tmp_path / "empty.dump"
        dump.write_text("")

        load_state_from_dump(system, str(dump))

        assert system.introduced == []
        assert system.time == pytest.approx(0.0)
        assert system.list_time == pytest.approx([0.0])
        assert system.topology_calls == [(set(), set())]

    def test_tolerance_controls_coordinate_matching(self, tmp_path, pzt_defects_dict):
        type_id = _type_id_for(pzt_defects_dict, "H")
        dump = write_dump(tmp_path / "tol.dump",
                          [(1, [atom(1, type_id, 0.5, 0.0, 0.0, 1.0)])])

        # 0.5 A away: accepted with a 1.0 A tolerance, rejected with 0.1 A.
        lenient = MockSystemState(make_grid(), pzt_defects_dict)
        load_state_from_dump(lenient, str(dump), tolerance=1.0)
        assert [entry["idx"] for entry in lenient.introduced] == [(0, 0, 0)]

        strict = MockSystemState(make_grid(), pzt_defects_dict)
        load_state_from_dump(strict, str(dump), tolerance=0.1)
        assert strict.introduced == []
