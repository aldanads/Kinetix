# tests/test_metadata_writer.py
"""
Behavioral spec for kinetix/utils/metadata.py (MetadataWriter).

Phase 1 of the simulator.py split: metadata/provenance writing extracted from
KMCSimulator.write_metadata / _get_git_provenance / _sanitize_numpy.

Real artifacts, no hardcoded config literals:
  - data/parameters/presets/VCM_mock.yaml        (SimulationConfig.from_yaml)
  - data/parameters/defects/VCM_HfO2_defects_config.yaml (DefectsConfig)
  - the species-id map is produced by the REAL production
    KMCSimulator._species_id_gen (on an uninitialized instance).

The MP summary is served from a pre-seeded local cache, so write_json never
touches the network. Only simulation_id (uuid) and timestamp_start are
nondeterministic; every other field is pinned against the pre-split payload.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from kinetix.configs.defect_config import DefectsConfig
from kinetix.configs.simulation_config import SimulationConfig
from kinetix.lattice.simulator import KMCSimulator
from kinetix.utils.metadata import MetadataWriter

PARAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "parameters"

# Offline stand-in for the Materials Project summary cache entry — the shape
# written by write_json itself on a cache miss (summary_minimal).
MP_SUMMARY = {
  "formula_pretty": "HfO2",
  "symmetry": {
    "crystal_system": "tetragonal",
    "symbol": "P4_2/nmc",
    "number": 137,
  },
}


# =============================================================================
# Fixtures — real parameter files through the production loaders
# =============================================================================

@pytest.fixture(scope="module")
def simulation_config() -> SimulationConfig:
  """REAL VCM_mock preset through the production loader."""
  return SimulationConfig.from_yaml(PARAMS_DIR / "presets" / "VCM_mock.yaml")


@pytest.fixture(scope="module")
def vcm_defects_dict() -> dict:
  """REAL VCM defects YAML through the production loader."""
  return DefectsConfig.from_yaml(
    PARAMS_DIR / "defects" / "VCM_HfO2_defects_config.yaml").to_dict()


def _make_crystal(tmp_path: Path, defects_dict: dict) -> KMCSimulator:
  """Uninitialized KMCSimulator carrying exactly the metadata attributes.

  Skips __init__/physics (same pattern as tests/test_state_loader.py); the MP
  summary cache is pre-seeded so write_json never touches the network.
  """
  crystal = KMCSimulator.__new__(KMCSimulator)
  crystal.defects_config = defects_dict
  crystal.id_material = "mp-offline-test"
  crystal.rank = 0
  crystal.api_key = None
  crystal.mpi_ctx = None
  cache_dir = tmp_path / "cache"
  cache_dir.mkdir(exist_ok=True)
  (cache_dir / f"summary_{crystal.id_material}.json").write_text(
    json.dumps(MP_SUMMARY))
  crystal.cache_dir = cache_dir
  crystal.chemical_formula = "HfO2"
  crystal.simulation_type = "electronic_device"
  crystal.temperature = 300.0
  crystal.miller_indices = (0, 0, 1)
  crystal.crystal_size = (3.0, 3.0, 3.0)
  crystal.partial_pressure = None
  crystal.sticking_coefficient = None
  crystal.time_step_limits = 1.0e-3
  crystal.n_search_superbasin = 0
  crystal.E_min = None
  return crystal


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
  out = tmp_path / "output"
  out.mkdir()
  return out


# =============================================================================
# Instantiation
# =============================================================================

def test_instantiation(simulation_config, output_dir):
  writer = MetadataWriter(output_dir, simulation_config)
  assert isinstance(writer.output_dir, Path)
  assert writer.output_dir == output_dir
  assert writer.config is simulation_config
  assert writer._git_provenance is None


def test_instantiation_coerces_str_output_dir(simulation_config, output_dir):
  writer = MetadataWriter(str(output_dir), simulation_config)
  assert isinstance(writer.output_dir, Path)


# =============================================================================
# write_json — payload compatible with the pre-split write_metadata
# =============================================================================

def test_write_json_default_filename(simulation_config, output_dir, tmp_path,
                                     vcm_defects_dict):
  crystal = _make_crystal(tmp_path, vcm_defects_dict)
  MetadataWriter(output_dir, simulation_config).write_json(crystal)
  assert (output_dir / "metadata.json").exists()


def test_write_json_custom_filename(simulation_config, output_dir, tmp_path,
                                    vcm_defects_dict):
  crystal = _make_crystal(tmp_path, vcm_defects_dict)
  MetadataWriter(output_dir, simulation_config).write_json(crystal, "custom.json")
  assert (output_dir / "custom.json").exists()
  assert not (output_dir / "metadata.json").exists()


def test_write_json_payload_matches_pre_split_format(simulation_config, output_dir,
                                                     tmp_path, vcm_defects_dict):
  crystal = _make_crystal(tmp_path, vcm_defects_dict)
  writer = MetadataWriter(output_dir, simulation_config)
  writer.write_json(crystal)

  data = json.loads((output_dir / "metadata.json").read_text())

  # Nondeterministic fields, verified separately below
  sim_id = data.pop("simulation_id")
  timestamp = data.pop("timestamp_start")

  assert re.fullmatch(r"HfO2_electronic_device_300K_[0-9a-f]{8}", sim_id)
  datetime.fromisoformat(timestamp)  # valid ISO-8601

  expected = {
    "metadata_version": "2.0",
    "workflow": {
      "type": "kinetic_monte_carlo",
      "code_name": "Kinetix",
      "code_url": "https://github.com/aldanads/Kinetix",
      "git": writer._get_git_provenance(),  # cached during write_json
    },
    "system": {
      "chemical_formula": "HfO2",
      "species": list(crystal.SPECIES_TYPE_MAP.keys()),
      "species_mapping": crystal.SPECIES_TYPE_MAP,
      "crystal_system": "tetragonal",
      "space_group": "P4_2/nmc",
      "lattice_type": 137,
      "film_orientation": "(0, 0, 1)",
      "growth_direction": [0, 0, 1],
      "simulation_domain_angstrom": [3.0, 3.0, 3.0],
      "periodic_boundary_conditions": [True, True, True],
      "materials_project_id": "mp-offline-test",
    },
    "conditions": {
      "simulation_type": "electronic_device",
      "temperature_K": 300.0,
      "partial_pressure_Pa": None,
      "sticking_coefficient": None,
      "simulation_time_limit_s": 1.0e-3,
    },
    "energy_model": {
      "source": "DFT-derived parameters",
      "superbasin_enabled": False,
      "superbasin_search_interval": 0,
      "superbasin_energy_threshold_ev": None,
    },
    "provenance": {
      "associated_publication_doi": "unpublished",
      "arxiv_id": "unpublished",
      "github_repository": "unpublished",
      "zenodo_concept_doi": "unpublished",
      "simulation_creator": "Samuel Aldana Delgado",
      "affiliation": "Tyndall National Institute, University College Cork",
      "funding": "TBD",
    },
  }
  assert data == expected


# =============================================================================
# Git provenance
# =============================================================================

def test_git_provenance_structure(simulation_config, output_dir):
  git = MetadataWriter(output_dir, simulation_config)._get_git_provenance()
  assert set(git) == {"commit", "branch", "is_clean"}
  assert isinstance(git["commit"], str)
  assert isinstance(git["branch"], str)
  assert isinstance(git["is_clean"], bool)


def test_git_provenance_cached(simulation_config, output_dir):
  writer = MetadataWriter(output_dir, simulation_config)
  first = writer._get_git_provenance()
  second = writer._get_git_provenance()
  assert first is second
  assert writer._git_provenance is first


# =============================================================================
# Future formats (documented, deliberately not implemented)
# =============================================================================

def test_write_h5md_not_implemented(simulation_config, output_dir):
  writer = MetadataWriter(output_dir, simulation_config)
  with pytest.raises(NotImplementedError, match="H5MD"):
    writer.write_h5md(object())


def test_push_to_nomad_not_implemented(simulation_config, output_dir):
  writer = MetadataWriter(output_dir, simulation_config)
  with pytest.raises(NotImplementedError, match="NOMAD"):
    writer.push_to_nomad(output_dir / "trajectory.h5")