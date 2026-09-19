"""Active learning helpers for MACE barrier validation and DFT feedback loops.

Every barrier computed with the MACE CI-NEB adapter is classified against a
set of physical sanity criteria (convergence, barrier magnitude, energy
profile shape, endpoint relaxation drift). Barriers that raise a flag are
exported -- together with enough metadata and structure information to
re-run them with DFT -- into an "active learning queue" directory. A DFT
workflow can then pick up the queue (manifest.json + one folder per barrier),
recompute the barriers at the higher level of theory, and the results are
used to refine the MACE model.

Typical use (see tests/test_mace_adapter.py::TestMACEAdapterAllPathways)::

    result = adapter.get_barrier(grid, o, d, use_cache=True, full_output=True)
    flags, priority = classify_barrier(result, endpoint_displacements)
    if flags:
        barrier_id = export_barrier_for_active_learning(...)
    create_active_learning_manifest(export_dir, problematic_barriers)

All messages go through the ``kinetix`` logger hierarchy, so they respect
``kinetix.logging_config.setup_logging()`` (terminal + optional log file).
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import write

logger = logging.getLogger(__name__)

# --- Classification thresholds ------------------------------------------------
BARRIER_TOO_HIGH = 5.0  # eV
BARRIER_TOO_LOW = 0.1   # eV
ENDPOINT_DISPLACEMENT_THRESHOLD = 0.5  # Angstrom
PROFILE_DIP_THRESHOLD = -0.3  # eV (relative to IS)


def classify_barrier(result, endpoint_displacements=None):
    """Classify a barrier calculation and return a list of problem flags.

    Parameters
    ----------
    result : dict
        Full output from ``mace_adapter.get_barrier(full_output=True)``.
        Cache-hit results carry only ``barrier``/``converged``/``profile``;
        freshly computed ones also carry ``endpoint_displacements`` and the
        NEB band, so pass ``endpoint_displacements`` explicitly when known.
    endpoint_displacements : dict, optional
        Dict with 'IS' and 'FS' displacement values in Angstrom.

    Returns
    -------
    flags : list of str
        List of problem flags (empty if barrier is OK).
    priority : str
        'high', 'medium', 'low', or 'none'.
    """
    flags = []

    # Check convergence
    if not result.get('converged', False):
        flags.append('not_converged')

    # Check barrier value (absent on failed calculations -> nothing to judge)
    barrier = result.get('barrier')
    if barrier is not None:
        if barrier > BARRIER_TOO_HIGH:
            flags.append(f'barrier_too_high ({barrier:.2f} eV)')
        elif barrier < BARRIER_TOO_LOW:
            flags.append(f'barrier_too_low ({barrier:.2f} eV)')

    # Check energy profile: the band should never dip well below the IS
    # energy (that signals a badly interpolated path or a mis-identified
    # transition state).
    profile = result.get('profile') or []
    if profile:
        min_rel_energy = min(profile)
        if min_rel_energy < PROFILE_DIP_THRESHOLD:
            flags.append(f'unphysical_profile (min={min_rel_energy:.2f} eV)')

    # Check endpoint relaxation: a migrating atom that drifts far from its
    # site during the pre-NEB relaxation means the site is not a real minimum
    # (wrong site guess, spontaneous defect rearrangement, ...).
    if endpoint_displacements:
        for endpoint, disp in endpoint_displacements.items():
            if disp > ENDPOINT_DISPLACEMENT_THRESHOLD:
                flags.append(
                    f'{endpoint}_displacement_too_large ({disp:.2f} A)')

    # Determine priority: how urgently does this barrier need DFT?
    if 'not_converged' in flags or any('barrier_too_high' in f for f in flags):
        priority = 'high'
    elif any('unphysical_profile' in f for f in flags):
        priority = 'medium'
    elif flags:
        priority = 'low'
    else:
        priority = 'none'

    return flags, priority


def format_barrier_info(origin_idx, dest_idx, result, distance, specie,
                        wall_time=None, endpoint_displacements=None):
    """Format barrier calculation info as a multi-line human-readable block.

    Returns
    -------
    str : Formatted multi-line string with all relevant information,
          including the active-learning classification.
    """
    lines = []
    lines.append(f"Hop: {origin_idx} -> {dest_idx}")
    lines.append(f"  Species: {specie}")
    lines.append(f"  Distance: {distance:.2f} A")

    barrier = result.get('barrier')
    if barrier is not None:
        lines.append(f"  Barrier: {barrier:.3f} eV")

    if 'converged' in result:
        lines.append(f"  Converged: {result['converged']}")

    profile = result.get('profile') or []
    if profile:
        lines.append(f"  Profile (rel to IS): {[f'{e:.2f}' for e in profile]}")
        lines.append(f"  Max along band: {max(profile):.3f} eV")
        lines.append(f"  Min along band: {min(profile):.3f} eV")

    if wall_time is not None:
        lines.append(f"  Wall time: {wall_time:.1f} s")

    if endpoint_displacements:
        lines.append("  Endpoint displacements:")
        for endpoint, disp in endpoint_displacements.items():
            lines.append(f"    {endpoint}: {disp:.3f} A")

    # Classification
    flags, priority = classify_barrier(result, endpoint_displacements)
    if flags:
        lines.append(f"  Flags: {', '.join(flags)}")
        lines.append(f"  Priority: {priority}")
    else:
        lines.append("  Status: OK")

    return '\n'.join(lines)


def _index_to_jsonable(idx):
    """Grid indices can be tuples of numpy integers; make them JSON-safe."""
    if isinstance(idx, (tuple, list)):
        return [_index_to_jsonable(i) for i in idx]
    if isinstance(idx, np.integer):
        return int(idx)
    if isinstance(idx, np.floating):
        return float(idx)
    return idx


def _classify_hop_type(origin_site, dest_site):
    """Classify the type of hop based on site types."""
    origin_type = origin_site.site_type
    dest_type = dest_site.site_type

    if origin_type == 'interstitial' and dest_type == 'interstitial':
        return 'oxygen_interstitial_migration'
    elif origin_type == 'O' and dest_type == 'O':
        return 'oxygen_vacancy_migration'
    else:
        return f'{origin_type}_to_{dest_type}'


def export_barrier_for_active_learning(
    mace_adapter, grid, origin_idx, dest_idx, result,
    output_dir, material_name, phase, barrier_counter,
    endpoint_displacements=None, label=None,
):
    """Export a barrier calculation for DFT validation and active learning.

    One folder per barrier is created under ``output_dir``::

        <output_dir>/<barrier_id>/
            IS.extxyz            unrelaxed initial-state cluster
            FS.extxyz            unrelaxed final-state cluster
            neb_band.extxyz      relaxed CI-NEB band (when available)
            metadata.json        full provenance + classification
            classification.json  flags/priority only (quick screening)

    Parameters
    ----------
    mace_adapter : KinetixMACEAdapter
        The adapter that computed the barrier.
    grid : dict
        The crystal grid.
    origin_idx, dest_idx : tuple
        Origin and destination site indices.
    result : dict
        Full output from ``get_barrier(full_output=True)``.
    output_dir : Path
        Base directory for active learning exports.
    material_name : str
        e.g., 'HfO2'.
    phase : str
        e.g., 'monoclinic'.
    barrier_counter : int
        Sequential counter for this barrier.
    endpoint_displacements : dict, optional
        Dict with 'IS' and 'FS' displacement values.
    label : str, optional
        Sweep label (e.g. 'interstitial' / 'vacancy') woven into the
        barrier_id so concurrent sweeps sharing one output_dir never
        overwrite each other's folders.

    Returns
    -------
    barrier_id : str
        The unique identifier for this barrier
        (e.g., 'HfO2_monoclinic_interstitial_001').
    """
    # Create barrier ID
    if label:
        barrier_id = f"{material_name}_{phase}_{label}_{barrier_counter:03d}"
    else:
        barrier_id = f"{material_name}_{phase}_{barrier_counter:03d}"
    barrier_dir = Path(output_dir) / barrier_id
    barrier_dir.mkdir(parents=True, exist_ok=True)

    # Build IS and FS structures (same construction path as get_barrier)
    start, end, frozen = mace_adapter.build_pair(grid, origin_idx, dest_idx)

    # Compute metadata
    origin_pos = np.array(grid[origin_idx].position, float)
    dest_pos = np.array(grid[dest_idx].position, float)
    v = mace_adapter.kx._minimum_image_vector(dest_pos - origin_pos)
    distance = float(np.linalg.norm(v))

    flags, priority = classify_barrier(result, endpoint_displacements)



    metadata = {
        'barrier_id': barrier_id,
        'material': material_name,
        'phase': phase,
        'sweep_label': label,
        'origin_idx': _index_to_jsonable(origin_idx),
        'dest_idx': _index_to_jsonable(dest_idx),
        'origin_pos': origin_pos.tolist(),
        'dest_pos': dest_pos.tolist(),
        'hop_distance': distance,
        'hop_type': _classify_hop_type(grid[origin_idx], grid[dest_idx]),
        'barrier_mace': result.get('barrier'),
        'converged': result.get('converged'),
        'profile': result.get('profile'),
        'n_atoms': result.get('n_atoms'),
        'wall_time': result.get('wall_time'),
        'model_id': mace_adapter.neb.model_id,
        'cluster_R_active': mace_adapter.neb.cluster['R_active'],
        'cluster_R_shell': mace_adapter.neb.cluster['R_shell'],
        'n_images': mace_adapter.neb.n_images,
        'fmax': mace_adapter.neb.fmax,
        'date_computed': datetime.now().isoformat(),
        'flags': flags,
        'priority': priority,
    }

    # The endpoint-displacement dict stays in metadata.json only: ASE's
    # extxyz writer cannot serialize nested dicts, so the atoms .info gets
    # one scalar per endpoint instead (disp_IS / disp_FS).
    if endpoint_displacements:
        metadata['endpoint_displacements'] = {
            k: float(v) for k, v in endpoint_displacements.items()}

    # info dict carried on the structures (extxyz-safe values only)
    info_meta = {k: v for k, v in metadata.items()
                 if k != 'endpoint_displacements'}
    if endpoint_displacements:
        for endpoint, disp in endpoint_displacements.items():
            info_meta[f'disp_{endpoint}'] = float(disp)

    # Add metadata to structures
    start.info.update(info_meta)
    end.info.update(info_meta)

    # Write structures
    write(barrier_dir / 'IS.extxyz', start, format='extxyz')
    write(barrier_dir / 'FS.extxyz', end, format='extxyz')

    # Write NEB band if available (relaxed endpoints + intermediate images)
    if result.get('neb_band'):
        band_atoms = result['neb_band']
        for i, img in enumerate(band_atoms):
            img.info.update(info_meta)
            img.info['image_index'] = i
        write(barrier_dir / 'neb_band.extxyz', band_atoms, format='extxyz')

    # Write metadata
    with open(barrier_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Write classification summary
    classification = {
        'barrier_id': barrier_id,
        'flags': flags,
        'priority': priority,
    }
    with open(barrier_dir / 'classification.json', 'w') as f:
        json.dump(classification, f, indent=2)

    logger.info("Exported barrier %s (priority=%s, flags=%s) to %s",
                barrier_id, priority, flags or ['none'], barrier_dir)

    return barrier_id


def create_active_learning_manifest(output_dir, barriers):
    """Create (or update) a manifest.json summarizing the exported barriers.

    The manifest is the entry point for the DFT validation workflow. If a
    manifest already exists in ``output_dir`` the new barriers are MERGED
    into it (deduplicated by barrier_id, newest entry wins), so successive
    sweeps -- e.g. the interstitial and vacancy tests sharing one queue
    directory -- accumulate into a single queue instead of clobbering
    each other.

    Parameters
    ----------
    output_dir : Path
        Directory containing the barrier folders.
    barriers : list of dict
        List of barrier summary dicts (as returned per-barrier by
        ``export_barrier_for_active_learning`` consumers).

    Returns
    -------
    Path : path to the written manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.json'

    existing = []
    if manifest_path.is_file():
        try:
            with open(manifest_path) as f:
                existing = json.load(f).get('barriers', [])
        except (json.JSONDecodeError, OSError, AttributeError):
            logger.warning("Could not parse existing manifest %s; "
                           "starting a fresh one", manifest_path)
            existing = []

    merged = {b['barrier_id']: b for b in existing}
    for b in barriers:
        merged[b['barrier_id']] = b
    merged_barriers = sorted(merged.values(), key=lambda b: b['barrier_id'])

    manifest = {
        'created': datetime.now().isoformat(),
        'total_barriers': len(merged_barriers),
        'barriers': merged_barriers,
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s (%d barriers)",
                manifest_path, len(merged_barriers))
    return manifest_path
