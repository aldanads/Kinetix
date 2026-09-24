# tests/test_mace_adapter.py
"""Pytest integration tests for the MACE CI-NEB adapter (KinetixMACEAdapter).

Run from the repository root::

    pytest tests/test_mace_adapter.py -m "not slow" -v

The ``mace_adapter`` fixture builds the adapter exactly like the production
pipeline does: the calculator settings (model source, model filename, device,
NEB protocol, cluster radii, cache dir) are read from the ``VCM_mock.yaml``
preset via ``SimulationConfig.from_yaml()`` and forwarded to
``KinetixMACEAdapter``. When the preset's ``calculator.model`` is a Hugging
Face repo ID, the model is fetched through ``MACENEBBarrierCalculator`` (and
cached under ``cache_dir/hf/``); when it is a local ``.model`` path, that file
is used directly. Both paths must work.

Device handling is cluster-ready: the preset may request ``device: "cuda"``;
on machines without a GPU the fixture falls back to ``"cpu"`` with a warning.

The comprehensive pathway sweeps (``TestMACEAdapterAllPathways``) additionally
classify every computed barrier with the active-learning helpers in
``kinetix/calculators/active_learning.py`` and export the suspicious ones
(convergence problems, implausible barrier values, unphysical profiles, large
endpoint relaxation drift) to ``test_output/active_learning_queue/`` as a DFT
validation queue (IS/FS structures + NEB band + metadata + manifest.json).

Skips cleanly when the optional mace-torch stack (torch + mace) is
unavailable, when the configured model cannot be resolved, or when the
Hugging Face Hub is unreachable (for the download test).
"""
import csv
import logging
import os
import socket
import sys
import time
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# These tests need the optional mace-torch stack (torch + mace) and either a
# GPU or minutes-to-hours of CPU NEB time. The pathway sweeps additionally
# carry @pytest.mark.slow (skipped unless --runslow, see tests/conftest.py).
# Deselect with: pytest tests/ -m "not mace"
pytestmark = pytest.mark.mace

# Repository root (parent of tests/); the model cache lives under data/cache/...
REPO_ROOT = Path(__file__).resolve().parent.parent

# Logger under the 'kinetix' hierarchy so kinetix.logging_config.setup_logging()
# (enabled by the sweep fixture below) routes the messages to the terminal.
logger = logging.getLogger("kinetix.tests.mace_adapter")

# --- Module-level import guards ---------------------------------------------
# Import the adapter from its submodule (kinetix/calculators/__init__.py is an
# empty marker package and intentionally does not pull the mace_neb dependency
# chain). If any import-time dependency (e.g. ASE) is missing, skip the whole
# module cleanly instead of failing collection.
try:
    from kinetix.configs.simulation_config import SimulationConfig
    from kinetix.initialization import initialization
    from kinetix.calculators.mace_neb import (KinetixMACEAdapter,
                                              MACENEBBarrierCalculator)
    from kinetix.calculators.active_learning import (
        classify_barrier,
        create_active_learning_manifest,
        export_barrier_for_active_learning,
        format_barrier_info,
    )
except ImportError as exc:  # pragma: no cover - only hit when deps are absent
    pytest.skip(f"MACE adapter imports unavailable: {exc}",
                allow_module_level=True)

# --- Constants ---------------------------------------------------------------
CONFIG_NAME = "VCM_mock.yaml"
PRESET_PATH = REPO_ROOT / "data" / "parameters" / "presets" / CONFIG_NAME
# Existing local model kept as the offline/local-path fallback: the preset may
# point at a HF repo ID or at this file; both code paths must work.
MODEL_PATH = (REPO_ROOT / "data" / "cache" / "neb_cache"
              / "HfO2_mh1_F_LONG_cpu.model")
# Production HF target, used by the fetch tests when the preset itself points
# at a local file instead of a repo ID.
HF_DEFAULT_REPO_ID = "jamesh12345/hfo2-mace"
HF_DEFAULT_FILENAME = "HfO2_mh1_F_LONG.model"
R_ACTIVE = 5.0
R_SHELL = 7.0
BARRIER_BOUNDS = (0.05, 5.0)  # eV
CACHE_MAX_S = 0.1             # seconds budget for a cached barrier lookup


# --- Preset-driven helpers (shared by fixtures and tests) --------------------
def _resolve_device(yaml_device):
    """Effective torch device for a ``calculator.device`` preset value.

    The preset is the source of truth. A CUDA request is validated against
    the actual hardware: if CUDA is unavailable (e.g. CPU-only laptop), the
    helper falls back to ``"cpu"`` and emits a RuntimeWarning so the run is
    still valid on GPU clusters without any code change. An empty value
    auto-selects the best available device.
    """
    requested = str(yaml_device or "").strip().lower()
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if not requested:  # unset -> best available device
        return "cuda" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        warnings.warn(
            f"calculator.device='{yaml_device}' requests CUDA but "
            "torch.cuda.is_available() is False on this machine; "
            "falling back to 'cpu'",
            RuntimeWarning,
        )
        return "cpu"
    return requested


def _is_local_model_path(model_value):
    """True when ``calculator.model`` refers to a local file (existing file
    anywhere, or a ``*.model`` path that is simply missing right now)."""
    value = str(model_value).strip()
    if not value:
        return False
    p = Path(value)
    if p.is_absolute():
        return p.is_file()
    return (REPO_ROOT / value).is_file() or value.endswith(".model")


def _hf_repo_and_filename(calculator_config):
    """(repo_id, filename) the HF fetch tests should exercise.

    Taken from the preset when it points at a HF repo (single source of
    truth for production); otherwise the known production target is used so
    the HF plumbing stays tested even with a local-path preset.
    """
    model = str(calculator_config.model).strip()
    if not _is_local_model_path(model):
        return model, str(calculator_config.model_filename or "model.model")
    return HF_DEFAULT_REPO_ID, HF_DEFAULT_FILENAME


def _hf_network_available(timeout=5.0):
    """Cheap reachability probe so offline machines skip download tests."""
    try:
        with socket.create_connection(("huggingface.co", 443),
                                      timeout=timeout):
            return True
    except OSError:
        return False


def _model_load_failure(mace_adapter):
    """Reason string when the MACE model cannot be deserialized on the
    resolved device, or None when it loads.

    A model saved on a GPU node embeds CUDA storages; a CPU-only torch build
    cannot restore them (NotImplementedError on 'aten::empty_strided' from
    the CUDA backend). On GPU clusters the same file loads fine, and the
    bundled CPU-saved fallback model works on CPU-only machines.
    """
    try:
        mace_adapter.neb._new_image_calculator()
        return None
    except NotImplementedError as exc:
        return (f"MACE model '{mace_adapter.neb.model_path}' cannot be "
                f"loaded on device '{mace_adapter.neb.device}': {exc}. "
                "A CUDA-saved model cannot be deserialized by a CPU-only "
                "torch build; on CPU-only machines use a CPU-saved model "
                "(e.g. the bundled HfO2_mh1_F_LONG_cpu.model), on GPU "
                "clusters the HF model loads with device='cuda'.")
    except Exception as exc:  # any other load failure -> skip with context
        return (f"MACE model '{mace_adapter.neb.model_path}' failed to "
                f"load: {type(exc).__name__}: {exc}")


# --- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope="module")
def calculator_config():
    """CalculatorConfig parsed from the production VCM_mock.yaml preset.

    Reading the preset (instead of hardcoding values) keeps the tests on the
    same code path as production runs, including HF repo IDs, the device
    setting and the NEB protocol.
    """
    config = SimulationConfig.from_yaml(PRESET_PATH)
    if config.calculator is None:
        pytest.skip(f"No 'calculator' section in {PRESET_PATH}")
    if config.calculator.type != "mace_neb":
        pytest.skip(f"Preset calculator type is "
                    f"'{config.calculator.type}', not 'mace_neb'")
    return config.calculator


@pytest.fixture(scope="module")
def model_source(calculator_config):
    """model_source exactly as production would consume it.

    Relative local paths are resolved against the repository root so the
    tests are CWD-independent; anything that is not an existing local file
    is passed through untouched (MACENEBBarrierCalculator then treats it as
    a Hugging Face repo ID and downloads the model).
    """
    raw = str(calculator_config.model).strip()
    p = Path(raw)
    if not p.is_absolute():
        rooted = REPO_ROOT / raw
        if rooted.is_file():
            return str(rooted)
        if raw.endswith(".model"):
            pytest.skip(f"Local MACE model not found: {raw} "
                        f"(looked for {rooted})")
    else:
        if p.is_file():
            return str(p)
        pytest.skip(f"Local MACE model not found: {p}")

    # Not an existing local file -> HF repo ID path; requires huggingface_hub.
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        pytest.skip(f"model '{raw}' is a HF repo ID but huggingface_hub "
                    "is not installed")
    return raw


@pytest.fixture(scope="module")
def resolved_cache_dir(calculator_config):
    """Preset cache_dir made absolute against the repository root."""
    cache_dir = Path(calculator_config.cache_dir or "data/cache/neb_cache")
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    return cache_dir


@pytest.fixture(scope="module")
def system_state():
    """KMC simulator built from the VCM mock preset (once per session)."""
    sim_id = 0
    params = {
        "vo_initial_concentration": 1.0e-2,
        "temperature": 293.0,
        "h_generation": 0.45,
    }
    simulator, *_ = initialization(sim_id, params, CONFIG_NAME)
    return simulator


@pytest.fixture(scope="module")
def mace_adapter(system_state, calculator_config, model_source,
                 resolved_cache_dir):
    """KinetixMACEAdapter wired from the production preset config.

    Exercises the same construction path as production (HF download logic
    included). CUDA requests fall back to CPU (with a warning) on machines
    without a GPU; a missing mace-torch stack skips cleanly.
    """
    device = _resolve_device(calculator_config.device)

    try:
        adapter = KinetixMACEAdapter(
            model_source,
            kx=system_state,
            model_filename=calculator_config.model_filename,
            device=device,
            default_dtype=calculator_config.default_dtype,
            cluster=calculator_config.cluster,
            cache_dir=str(resolved_cache_dir),
            n_images=calculator_config.n_images,
            fmax=calculator_config.fmax,
            max_steps=calculator_config.max_steps,
        )
    except ImportError as exc:
        pytest.skip(f"MACE dependencies unavailable: {exc}")

    print(f"mace_adapter: model_source={model_source} "
          f"model_filename={calculator_config.model_filename} "
          f"device={device} cluster={calculator_config.cluster} "
          f"n_images={calculator_config.n_images} "
          f"fmax={calculator_config.fmax}")
    return adapter


@pytest.fixture(scope="module")
def oi_hop(system_state):
    """(origin_idx, dest_idx) for an oxygen interstitial hop well inside the
    domain (>= R_SHELL from the z boundaries). Built once per session."""
    lattice = system_state.structure.lattice
    # Geometric center of the supercell (orientation-independent).
    center = lattice.get_cartesian_coords([0.5, 0.5, 0.5])

    # Fractional threshold: convert R_SHELL (Å) to fractional z units.
    # Uses the z-component of the c-vector (the film-normal direction).
    z_height = abs(lattice.matrix[2][2])
    frac_thr = R_SHELL / z_height

    # Empty interstitial closest to the geometric center, away from the
    # z boundaries, so the cluster cut sees bulk-like surroundings.
    best, origin_idx = np.inf, None
    for idx, site in system_state.grid_crystal.items():
        if site.site_type != "interstitial" or site.defect.chemical_specie != "Empty":
            continue

        # Orientation-independent z-boundary check (replaces Cartesian z
        # vs crystal_size[2]). Site must be >= R_SHELL from top and bottom.
        frac = lattice.get_fractional_coords(site.position)
        if not (frac_thr <= frac[2] <= 1.0 - frac_thr):
            continue
        d = np.linalg.norm(np.array(site.position) - center)
        if d < best:
            best, origin_idx = d, idx

    if origin_idx is None:
        pytest.skip("No bulk-like empty interstitial found in the mock grid")

    # Introduce an oxygen interstitial at the origin site.
    cfg = system_state.defects_config["oxygen_interstitial"]
    support_update_sites = set()
    event_update_sites = set()
    system_state._introduce_specie_site(
        origin_idx, support_update_sites, event_update_sites,
        cfg["symbol"], cfg["charge"],
    )

    # Destination: a neighboring empty interstitial.
    origin = system_state.grid_crystal[origin_idx]
    dest_idx = None
    for n in origin.nearest_neighbors_idx:
        neighbor = system_state.grid_crystal[n]
        if neighbor.site_type == "interstitial" and neighbor.defect.chemical_specie == "Empty":
            dest_idx = n
            break

    # Fallback: search the whole grid for the nearest empty interstitial
    if dest_idx is None:
        origin_pos = np.array(origin.position)
        best_dist = np.inf
        for idx, neighbor in system_state.grid_crystal.items():
            if idx == origin_idx:
                continue
            if neighbor.site_type != "interstitial":
                continue
            if neighbor.defect.chemical_specie != "Empty":
                continue
            # Use minimum image distance
            v = system_state._minimum_image_vector(np.array(neighbor.position) - origin_pos)
            dist = np.linalg.norm(v)
            if dist < best_dist:
                best_dist = dist
                dest_idx = idx

        if dest_idx is None:
            pytest.skip("No neighboring empty interstitial found for the hop test")

    system_state.update_sites_topology(support_update_sites,
                                       event_update_sites)
    return origin_idx, dest_idx


# =============================================================================
# Geometry tests: pure adapter bookkeeping, no NEB run required.
# =============================================================================
class TestMACEAdapterGeometry:
    """Candidate-cluster search and IS/FS pair construction."""

    def test_candidate_keys_exact(self, mace_adapter, oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop

        # Hop center: midpoint between origin and destination.
        origin_pos = np.array(grid[origin_idx].position, float)
        dest_pos = np.array(grid[dest_idx].position, float)
        center = 0.5 * (origin_pos + dest_pos)

        # Brute-force expected set: every grid site whose minimum-image
        # distance to the center is <= R_SHELL.
        expected = {
            k for k, s in grid.items()
            if np.linalg.norm(
                mace_adapter.kx._minimum_image_vector(
                    np.array(s.position, float) - center)
            ) <= R_SHELL
        }

        found = set(mace_adapter._candidate_keys(center))
        assert expected == found, "candidate search mismatch"
        print(f"found {len(found)} candidate sites within "
              f"R_shell={R_SHELL} Angstrom")

    def test_build_pair_geometry(self, mace_adapter, oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop

        start, end, frozen = mace_adapter.build_pair(
            grid, origin_idx, dest_idx)

        assert len(start) == len(end)
        assert sorted(set(start.symbols)) == ["Hf", "O"]
        assert np.allclose(start.positions[-1], grid[origin_idx].position)
        assert np.allclose(end.positions[-1], grid[dest_idx].position)
        assert len(start) - 1 not in frozen, "moving atom must never be frozen"
        assert len(frozen) > 0, "expected a frozen shell in cluster mode"
        print(f"cluster: {len(start)} atoms "
              f"({len(frozen)} frozen in the R_Active={R_ACTIVE} shell)")

        """
        from ase.io import write
        write("IS.extxyz", start)
        write("FS.extxyz", end)
        """    

# =============================================================================
# Barrier tests: actual CI-NEB runs (need torch + mace-torch installed).
# =============================================================================
class TestMACEAdapterBarrier:
    """Barrier sanity/convergence and SQLite cache-hit performance.

    These tests actually run CI-NEB and therefore additionally need the
    optional mace-torch stack (torch + mace); without it they skip cleanly.
    """

    @pytest.fixture(scope="module")
    def requires_mace(self):
        """Skip the barrier tests cleanly when torch/mace-torch are missing."""
        try:
            import mace  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"mace-torch (torch + mace) not installed: {exc}")

    @pytest.fixture(scope="module")
    def loadable_model(self, requires_mace, mace_adapter):
        """Skip when the model cannot be deserialized on the resolved device
        (e.g. CUDA-saved HF model on a CPU-only torch build)."""
        reason = _model_load_failure(mace_adapter)
        if reason:
            pytest.skip(reason)

    def test_barrier_sane_and_converged(self, requires_mace, loadable_model,
                                        mace_adapter, oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop
        result = mace_adapter.get_barrier(grid, origin_idx, dest_idx,
                                          use_cache=True, full_output=True)

        print(f"Profile (eV, rel to IS): {result['profile']}")   # ← ADD THIS
        print(f"Max along band: {max(result['profile']):.4f} eV")

        assert result["converged"] is True
        assert (BARRIER_BOUNDS[0] < result["barrier"]
                < BARRIER_BOUNDS[1])
        print(f"O_i hop barrier: {result['barrier']:.3f} eV "
              f"(converged={result['converged']})")

    def test_cache_hit_fast(self, requires_mace, loadable_model, mace_adapter,
                            oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop

        # First call populates the SQLite cache.
        mace_adapter.get_barrier(grid, origin_idx, dest_idx,
                                 use_cache=True)

        t0 = time.perf_counter()
        mace_adapter.get_barrier(grid, origin_idx, dest_idx, use_cache=True)
        dt = time.perf_counter() - t0
        assert dt < CACHE_MAX_S, f"cache hit too slow: {dt:.4f}s"
        print(f"cache hit time: {dt:.4f} s")

    def test_device_propagation(self, requires_mace, loadable_model,
                                mace_adapter, calculator_config):
        """calculator.device survives config -> adapter -> MACE model.

        Checks the effective device recorded on the NEB calculator and the
        device the underlying torch model parameters actually live on.
        """
        import torch

        expected = _resolve_device(calculator_config.device)
        assert mace_adapter.neb.device == expected, (
            f"NEB calculator device '{mace_adapter.neb.device}' does not "
            f"match the preset-derived device '{expected}'")

        calc = mace_adapter.neb._new_image_calculator()
        model_device = next(calc.models[0].parameters()).device
        assert model_device.type == torch.device(expected).type, (
            f"MACE model is on '{model_device}', expected "
            f"'{torch.device(expected).type}'")
        print(f"MACE model device: {model_device} "
              f"(preset device={calculator_config.device}, "
              f"effective={expected})")

    def test_cuda_fallback_warns_without_gpu(self, requires_mace):
        """A 'cuda' request without a GPU warns and resolves to 'cpu'."""
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine; "
                        "no CPU fallback to exercise")
        with pytest.warns(RuntimeWarning, match="CUDA"):
            device = _resolve_device("cuda")
        assert device == "cpu"

    # def


# =============================================================================
# Model-fetch plumbing: MACENEBBarrierCalculator resolves local files and
# Hugging Face repo IDs (no NEB runs involved, so these are fast).
# =============================================================================
class TestMACEModelFetch:
    """Model resolution of MACENEBBarrierCalculator (local path vs HF repo)."""

    def test_local_model_load(self, tmp_path):
        """A local .model file is used directly, no HF machinery involved."""
        if not MODEL_PATH.is_file():
            pytest.skip(f"local MACE model not found: {MODEL_PATH}")

        calc = MACENEBBarrierCalculator(
            str(MODEL_PATH), cache_dir=str(tmp_path / "cache"), device="cpu")

        assert calc.model_path == MODEL_PATH
        assert calc.model_path.is_file()
        assert calc.model_id == MODEL_PATH.name
        # The cache directory was created and holds the barrier database.
        assert calc.cache_dir == tmp_path / "cache"
        assert calc.cache_dir.is_dir()
        assert (calc.cache_dir / "barrier.db").is_file()
        print(f"local model loaded: {calc.model_path}")

    def test_hf_model_download(self, calculator_config, resolved_cache_dir):
        """A HF repo ID is fetched via hf_hub_download into cache_dir/hf/.

        Uses the persistent production cache, so the first run downloads the
        model once and every later run is a fast cache hit.
        """
        pytest.importorskip("huggingface_hub")
        repo_id, filename = _hf_repo_and_filename(calculator_config)
        if not _hf_network_available():
            pytest.skip("Hugging Face Hub unreachable (offline environment)")

        calc = MACENEBBarrierCalculator(
            repo_id, cache_dir=str(resolved_cache_dir),
            model_filename=filename, device="cpu")

        model_path = calc.model_path
        assert model_path.is_file(), f"downloaded model missing: {model_path}"
        assert model_path.name == filename
        # Cached inside <cache_dir>/hf/ with the standard HF hub layout
        # (models--<org>--<repo>/snapshots/<revision>/<filename>).
        hf_branch = calc.cache_dir / "hf"
        assert hf_branch in model_path.parents, (
            f"{model_path} is not cached under {hf_branch}")
        assert any(part.startswith("models--") for part in model_path.parts)

        # A second construction must resolve to the identical cached
        # snapshot (no re-download).
        calc2 = MACENEBBarrierCalculator(
            repo_id, cache_dir=str(resolved_cache_dir),
            model_filename=filename, device="cpu")
        assert calc2.model_path == model_path
        print(f"HF model cached at: {model_path}")

    def test_hf_model_filename_parameter(self, tmp_path, calculator_config):
        """model_filename is forwarded to hf_hub_download verbatim, and a
        missing file in the repo raises a clear error."""
        pytest.importorskip("huggingface_hub")
        repo_id, filename = _hf_repo_and_filename(calculator_config)
        import huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError

        stub_model = tmp_path / "stub.model"
        stub_model.write_bytes(b"stub MACE model")

        with mock.patch.object(huggingface_hub, "hf_hub_download",
                               return_value=str(stub_model)) as dl:
            calc = MACENEBBarrierCalculator(
                repo_id, cache_dir=str(tmp_path / "cache"),
                model_filename=filename, device="cpu")

        dl.assert_called_once()
        kwargs = dl.call_args.kwargs
        assert kwargs["repo_id"] == repo_id
        assert kwargs["filename"] == filename
        assert Path(kwargs["cache_dir"]) == tmp_path / "cache" / "hf"
        assert calc.model_path == stub_model
        assert calc.model_id == stub_model.name
        print(f"hf_hub_download called with repo_id={kwargs['repo_id']}, "
              f"filename={kwargs['filename']}, cache_dir={kwargs['cache_dir']}")

        # A filename that does not exist in the repo -> clear error naming
        # the missing file (and the repo it was requested from).
        with mock.patch.object(
                huggingface_hub, "hf_hub_download",
                side_effect=EntryNotFoundError("no such file")):
            with pytest.raises(FileNotFoundError,
                               match="does_not_exist.model"):
                MACENEBBarrierCalculator(
                    repo_id, cache_dir=str(tmp_path / "cache2"),
                    model_filename="does_not_exist.model", device="cpu")

    def test_missing_huggingface_hub(self, tmp_path):
        """Without huggingface_hub: local paths still work, HF repo IDs
        raise a clear ImportError."""
        if MODEL_PATH.is_file():
            with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
                calc = MACENEBBarrierCalculator(
                    str(MODEL_PATH), cache_dir=str(tmp_path / "local"),
                    device="cpu")
            assert calc.model_path == MODEL_PATH, (
                "local model path must not require huggingface_hub")

        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                MACENEBBarrierCalculator(
                    HF_DEFAULT_REPO_ID, cache_dir=str(tmp_path / "hf"),
                    device="cpu")


# =============================================================================
# Comprehensive pathway tests: barriers for ALL migration pathways from
# representative bulk-like sites (SLOW: many CI-NEB calculations, hours).
# Run with: pytest tests/test_mace_adapter.py -k AllPathways -v
# Skip with: pytest tests/test_mace_adapter.py -m "not slow"
# =============================================================================
def _representative_site(system_state, wanted_types, wanted_specie=None):
    """Site of one of wanted_types closest to the supercell center, away from
    the z boundaries (>= R_SHELL fractional check, as in oi_hop)."""
    lattice = system_state.structure.lattice
    center = lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    z_height = abs(lattice.matrix[2][2])
    frac_thr = R_SHELL / z_height

    candidates = []
    for idx, site in system_state.grid_crystal.items():
        if site.site_type not in wanted_types:
            continue
        if wanted_specie is not None and site.defect.chemical_specie != wanted_specie:
            continue
        frac = lattice.get_fractional_coords(site.position)
        if not (frac_thr <= frac[2] <= 1.0 - frac_thr):
            continue
        d = np.linalg.norm(np.array(site.position, float) - center)
        candidates.append((d, idx))

    if not candidates:
        pytest.skip(f"No bulk-like {wanted_types} site found in the mock grid")

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


@pytest.mark.slow
class TestMACEAdapterAllPathways:
    """Comprehensive barrier calculations for ALL pathways from representative
    bulk-like sites. Each hop runs a full CI-NEB relaxation, so this class
    takes hours; deselect during rapid development with `pytest -m "not slow"`.
    """

    @pytest.fixture(scope="class")
    def representative_interstitial(self, system_state):
        """Bulk-like EMPTY interstitial closest to the supercell center."""
        return _representative_site(system_state, ("interstitial",), "Empty")

    @pytest.fixture(scope="class")
    def representative_vacancy(self, system_state):
        """Host oxygen site closest to the supercell center."""
        return _representative_site(system_state, ("O",), "O")

    @pytest.fixture(scope="class", autouse=True)
    def _terminal_logging(self):
        """Route sweep messages (INFO and up) to the terminal.

        pytest does not configure the 'kinetix' logger hierarchy by default,
        so logger.info() calls would be silently dropped. setup_logging()
        attaches the production stdout handler, keeping the per-hop progress
        visible in cluster logs while still honoring kinetix log levels.
        """
        from kinetix.logging_config import setup_logging
        setup_logging(level=logging.INFO)

    @staticmethod
    def _calculate_all_pathways(system_state, mace_adapter, origin_idx,
                                dest_site_types, csv_path, export_dir=None,
                                material_name="HfO2", phase="monoclinic",
                                sweep_label=None):
        """Barriers for all eligible neighbors of origin_idx (defect already
        introduced by the caller).

        Every barrier is classified with the active-learning helpers
        (convergence / barrier magnitude / profile shape / endpoint drift)
        and the flagged ones are exported to ``export_dir`` as a DFT
        validation queue. Results are written to ``csv_path``; the compact
        per-hop progress line is kept on the terminal so the evolution of
        the sweep is easy to follow in cluster logs.

        Returns (results, problematic_barriers).
        """
        grid = system_state.grid_crystal
        origin_site = grid[origin_idx]
        origin_pos = np.array(origin_site.position, float)

        results = []
        problematic_barriers = []
        barrier_counter = 1

        logger.info("=" * 80)
        logger.info("Calculating pathways from site %s (%s sweep)",
                    origin_idx, sweep_label or "generic")
        logger.info("=" * 80)

        for neighbor_idx in origin_site.nearest_neighbors_idx:
            neighbor_site = grid[neighbor_idx]
            if neighbor_site.site_type not in dest_site_types:
                continue

            # Distance under the minimum image convention.
            dest_pos = np.array(neighbor_site.position, float)
            v = system_state._minimum_image_vector(dest_pos - origin_pos)
            distance = np.linalg.norm(v)

            try:
                result = mace_adapter.get_barrier(
                    grid, origin_idx, neighbor_idx,
                    use_cache=True, full_output=True)
            except Exception as exc:  # non-converging NEB must not kill sweep
                logger.error("Hop %s -> %s: FAILED: %s",
                             origin_idx, neighbor_idx, exc)
                results.append({
                    "origin_idx": str(origin_idx),
                    "dest_idx": str(neighbor_idx),
                    "origin_pos": " ".join(f"{c:.4f}" for c in origin_pos),
                    "dest_pos": " ".join(f"{c:.4f}" for c in dest_pos),
                    "distance": f"{distance:.4f}",
                    "barrier": "FAILED",
                    "converged": False,
                    "chemical_specie": neighbor_site.defect.chemical_specie,
                    "flags": "exception_raised",
                    "priority": "high",
                })
                continue

            # Endpoint relaxation drift (only present on freshly computed
            # barriers; cache hits carry barrier/converged/profile only).
            endpoint_displacements = result.get("endpoint_displacements") or {}

            # Detailed multi-line info block plus the compact one-line
            # progress line (kept on purpose: it makes the evolution of the
            # sweep easy to follow in cluster logs).
            info_str = format_barrier_info(
                origin_idx, neighbor_idx, result, distance,
                neighbor_site.defect.chemical_specie,
                wall_time=result.get("wall_time"),
                endpoint_displacements=endpoint_displacements)
            print(info_str)
            barrier_str = (f"{result['barrier']:.3f} eV"
                           if result.get("barrier") is not None else "FAILED")
            print(f"Hop {origin_idx} -> {neighbor_idx}: "
                  f"origin_pos={origin_pos} final_pos={dest_pos} "
                  f"distance={distance:.3f} Ang barrier={barrier_str} "
                  f"specie={neighbor_site.defect.chemical_specie}")
            print()

            # Classify the barrier for the active-learning feedback loop.
            flags, priority = classify_barrier(result, endpoint_displacements)

            results.append({
                "origin_idx": str(origin_idx),
                "dest_idx": str(neighbor_idx),
                "origin_pos": " ".join(f"{c:.4f}" for c in origin_pos),
                "dest_pos": " ".join(f"{c:.4f}" for c in dest_pos),
                "distance": f"{distance:.4f}",
                "barrier": (f"{result['barrier']:.4f}"
                            if result.get("barrier") is not None else ""),
                "converged": result.get("converged", False),
                "chemical_specie": neighbor_site.defect.chemical_specie,
                "flags": ", ".join(flags) if flags else "OK",
                "priority": priority,
            })

            # Export flagged barriers for DFT validation / model refinement.
            if flags and export_dir:
                barrier_id = export_barrier_for_active_learning(
                    mace_adapter, grid, origin_idx, neighbor_idx, result,
                    export_dir, material_name, phase, barrier_counter,
                    endpoint_displacements=endpoint_displacements,
                    label=sweep_label)
                problematic_barriers.append({
                    "barrier_id": barrier_id,
                    "origin_idx": str(origin_idx),
                    "dest_idx": str(neighbor_idx),
                    "barrier_mace": result.get("barrier"),
                    "flags": flags,
                    "priority": priority,
                })
                barrier_counter += 1
                print(f"  -> Exported as {barrier_id}\n")

        # Write CSV
        if results:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            logger.info("Results saved to %s", csv_path)

        # Summary
        n_ok = sum(1 for r in results if r["flags"] == "OK")
        n_failed = sum(1 for r in results if r["barrier"] == "FAILED")
        n_problematic = len(results) - n_ok - n_failed

        logger.info("=" * 80)
        logger.info("Summary: %d pathways from site %s",
                    len(results), origin_idx)
        logger.info("  OK: %d", n_ok)
        logger.info("  Problematic (flagged): %d", n_problematic)
        logger.info("  Failed: %d", n_failed)
        if problematic_barriers:
            logger.info("  Exported to active learning queue: %d (%s)",
                        len(problematic_barriers), export_dir)
        logger.info("=" * 80)

        return results, problematic_barriers

    @pytest.mark.slow
    def test_all_interstitial_pathways(self, system_state, mace_adapter,
                                       representative_interstitial):
        """Barriers for ALL interstitial hops from a representative O_i site,
        with active-learning classification and DFT-queue export."""
        origin_idx = representative_interstitial

        # Introduce an oxygen interstitial at the representative site.
        cfg = system_state.defects_config["oxygen_interstitial"]
        support_update_sites = set()
        event_update_sites = set()
        system_state._introduce_specie_site(
            origin_idx, support_update_sites, event_update_sites,
            cfg["symbol"], cfg["charge"],
        )
        system_state.update_sites_topology(support_update_sites,
                                           event_update_sites)

        csv_path = REPO_ROOT / "test_output" / "interstitial_pathways.csv"
        export_dir = REPO_ROOT / "test_output" / "active_learning_queue"

        results, problematic = self._calculate_all_pathways(
            system_state, mace_adapter, origin_idx, ("interstitial",),
            str(csv_path), export_dir=export_dir,
            material_name="HfO2", phase="monoclinic",
            sweep_label="interstitial")

        assert len(results) > 0, "No interstitial neighbor pathways found"

        # Write/refresh the manifest for the DFT validation queue.
        if problematic:
            create_active_learning_manifest(export_dir, problematic)

    @pytest.mark.slow
    def test_all_vacancy_pathways(self, system_state, mace_adapter,
                                  representative_vacancy):
        """Barriers for ALL oxygen hops from a representative V_O site.

        V_O is represented by the host oxygen site becoming an Empty space;
        _introduce_specie_site with the oxygen_vacancy config marks it V_O and
        the MACE adapter then builds the IS/FS pair with the oxygen REMOVED
        (build_pair skips the vacant site's atom), so no special handling is
        needed here beyond the standard introduction.
        """
        origin_idx = representative_vacancy

        # Introduce an oxygen vacancy at the representative host site.
        cfg = system_state.defects_config["oxygen_vacancy"]
        support_update_sites = set()
        event_update_sites = set()
        system_state._introduce_specie_site(
            origin_idx, support_update_sites, event_update_sites,
            cfg["symbol"], cfg["charge"],
        )
        system_state.update_sites_topology(support_update_sites,
                                           event_update_sites)

        csv_path = REPO_ROOT / "test_output" / "vacancy_pathways.csv"
        export_dir = REPO_ROOT / "test_output" / "active_learning_queue"

        results, problematic = self._calculate_all_pathways(
            system_state, mace_adapter, origin_idx, ("O",), str(csv_path),
            export_dir=export_dir, material_name="HfO2", phase="monoclinic",
            sweep_label="vacancy")

        assert len(results) > 0, "No oxygen neighbor pathways found"

        # Write/refresh the manifest for the DFT validation queue.
        if problematic:
            create_active_learning_manifest(export_dir, problematic)