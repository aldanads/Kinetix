# -*- coding: utf-8 -*-
"""Simulation metadata output.

Phase 1 of the ``crystal.py`` split: metadata/provenance writing is fully
decoupled from the physics and extracted from ``Crystal_Lattice``.

Current format: JSON (``metadata.json`` consumed by the kMC analysis tools).
Planned formats: H5MD (NOMAD compatibility) and a direct NOMAD repository push.
"""
from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymatgen.ext.matproj import MPRester

if TYPE_CHECKING:
    from kinetix.configs.simulation_config import SimulationConfig
    from kinetix.lattice.crystal import Crystal_Lattice

logger = logging.getLogger(__name__)


class MetadataWriter:
    """Handles simulation metadata output in multiple formats.

    Current: JSON (for kMC analysis)
    Future: H5MD (for NOMAD), direct NOMAD repository push

    Args:
        output_dir: Directory the metadata file is written into.
        simulation_config: Configuration of the run. Stored for the future
            H5MD/NOMAD formats; the JSON payload is currently sourced
            entirely from the ``Crystal_Lattice`` passed to ``write_json``
            so the output stays byte-compatible with the pre-split
            ``Crystal_Lattice.write_metadata``.
    """

    def __init__(self, output_dir: Path, simulation_config: SimulationConfig) -> None:
        self.output_dir = Path(output_dir)
        self.config = simulation_config
        self._git_provenance: dict[str, Any] | None = None

    def write_json(self, crystal: Crystal_Lattice, filename: str = "metadata.json") -> None:
        """Write metadata as JSON.

        Extracted from ``Crystal_Lattice.write_metadata``; the payload layout
        must not change (backward compatibility with existing analyses).

        Args:
            crystal: Lattice state whose metadata is recorded.
            filename: File name (inside ``output_dir``) to write.
        """
        metadata_path = self.output_dir / filename
        crystal._species_id_gen()

        crystal_data = {}

        # === Try cache first ===
        cache_key = f'summary_{crystal.id_material}'
        summary_dict = crystal._load_mp_cache(cache_key)

        try:
            # === Cache miss: fetch from API (rank 0)
            if summary_dict is None:
                if crystal.rank == 0:
                    with MPRester(crystal.api_key) as mpr:
                        # Query summary endpoint (most efficient)
                        results = mpr.materials.summary.search(
                            material_ids=[crystal.id_material]
                        )
                        summary_dict = results[0] if results else None
                        if summary_dict:
                            summary_minimal = {
                                'formula_pretty': summary_dict.get('formula_pretty', 'Unknown'),
                                'symmetry': {
                                    'crystal_system': summary_dict.get('symmetry', {}).get('crystal_system', 'Unknown'),
                                    'symbol': summary_dict.get('symmetry', {}).get('symbol', 'Unknown'),
                                    'number': summary_dict.get('symmetry', {}).get('number', 0),
                                },
                            }
                            crystal._save_mp_cache(cache_key, summary_minimal)
                            summary_dict = summary_minimal
                else:
                    summary_dict = None

                if crystal.mpi_ctx is not None:
                    summary_dict = crystal.mpi_ctx.bcast(summary_dict, root=0)

            if summary_dict:
                symm = summary_dict.get('symmetry', {})
                crystal_data.update({
                    "crystal_system": symm.get('crystal_system', 'Unknown'),
                    "space_group": symm.get('symbol', 'Unknown'),
                    "space_group_number": symm.get('number', 0),
                })
                logger.info("MP data fetched for %s: %s (%s)", crystal.id_material,
                            crystal_data['space_group'], crystal_data['crystal_system'])
            else:
                logger.warning("MP query returned no results for %s", crystal.id_material)
                crystal_data.update({
                    "crystal_system": "Unknown",
                    "space_group": "Unknown",
                    "space_group_number": 0,
                })
        except Exception as e:
            logger.warning("MP query failed: %s", e)
            crystal_data.update({
                "crystal_system": "Unknown",
                "space_group": "Unknown",
                "space_group_number": 0,
            })

        git_metadata = self._get_git_provenance()

        sim_id = (f"{crystal.chemical_formula}_{crystal.simulation_type}_"
                  f"{int(crystal.temperature)}K_{uuid.uuid4().hex[:8]}")

        metadata = {
            "metadata_version": "2.0",
            "simulation_id": sim_id,
            "timestamp_start": datetime.now().isoformat(),

            "workflow": {
                "type": "kinetic_monte_carlo",
                "code_name": "Kinetix",
                "code_url": "https://github.com/aldanads/Kinetix",
                "git": git_metadata,
            },

            "system": {
                "chemical_formula": crystal.chemical_formula,
                "species": list(crystal.SPECIES_TYPE_MAP.keys()),
                "species_mapping": crystal.SPECIES_TYPE_MAP,
                "crystal_system": crystal_data["crystal_system"],
                "space_group": crystal_data["space_group"],
                "lattice_type": crystal_data["space_group_number"],
                "film_orientation": f"{crystal.miller_indices}",
                "growth_direction": [d for d in crystal.miller_indices],
                "simulation_domain_angstrom": self._sanitize_numpy(crystal.crystal_size),
                "periodic_boundary_conditions": [True, True, True],
                "materials_project_id": crystal.id_material,
            },

            "conditions": {
                'simulation_type': crystal.simulation_type,
                'temperature_K': float(crystal.temperature),
                'partial_pressure_Pa': float(crystal.partial_pressure) if crystal.partial_pressure is not None else None,
                'sticking_coefficient': float(crystal.sticking_coefficient) if crystal.sticking_coefficient is not None else None,
                'simulation_time_limit_s': self._sanitize_numpy(crystal.time_step_limits),
            },

            "energy_model": {
                "source": "DFT-derived parameters",
                "superbasin_enabled": crystal.n_search_superbasin > 0,
                "superbasin_search_interval": self._sanitize_numpy(crystal.n_search_superbasin),
                "superbasin_energy_threshold_ev": float(crystal.E_min) if crystal.E_min is not None else None,
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

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def write_h5md(self, crystal: Crystal_Lattice, filename: str = "trajectory.h5") -> None:
        """Write metadata as H5MD (future - for NOMAD compatibility)."""
        raise NotImplementedError("H5MD support planned for future phase")

    def push_to_nomad(self, h5md_path: Path) -> None:
        """Push H5MD file to NOMAD repository (future)."""
        raise NotImplementedError("NOMAD integration planned for future phase")

    def _get_git_provenance(self) -> dict:
        """Get git commit hash, branch, and dirty status (cached)."""
        if self._git_provenance is None:
            self._git_provenance = self._compute_git_provenance()
        return self._git_provenance

    def _compute_git_provenance(self) -> dict:
        """Compute git provenance (moved from ``Crystal_Lattice._get_git_provenance``).

        Returns:
            dict: ``{"commit": ..., "branch": ..., "is_clean": ...}``. If git is
            unavailable (not installed, not a repository, or any other failure),
            the placeholders ``"unknown"`` / ``False`` are returned and a warning
            is logged instead of raising, so metadata writing never aborts a run.
        """
        git_info: dict[str, Any] = {"commit": "unknown", "branch": "unknown", "is_clean": False}
        try:
            git_info["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            try:
                branch = subprocess.check_output(
                    ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
                ).decode().strip()
            except subprocess.CalledProcessError:
                branch = subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
                ).decode().strip()
                if branch == "HEAD":
                    branch = "detached_HEAD"
            git_info["branch"] = branch if branch else "unknown"

            status = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info["is_clean"] = len(status) == 0
        except (subprocess.CalledProcessError, OSError) as e:
            # OSError covers FileNotFoundError (git not installed) and
            # PermissionError; CalledProcessError covers "not a git repo".
            # Keep the placeholder values and warn — never abort the run.
            logger.warning(
                "Git provenance unavailable (%s: %s); recording 'unknown' for commit/branch.",
                type(e).__name__, e,
            )
        return git_info

    @staticmethod
    def _sanitize_numpy(value):
        """Safely convert numpy types to native Python types for JSON serialization."""
        if hasattr(value, 'item'):      # numpy scalar (e.g., np.float64)
            return value.item()
        if hasattr(value, 'tolist'):    # numpy array
            return value.tolist()
        return value