# kinetix/calculators/mace_neb.py
"""mace_neb.py — MACE CI-NEB barrier calculator with model & barrier caching."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import get_distances
from ase.mep import NEB
from ase.optimize import FIRE

from kinetix.calculators.base import ActivationEnergyCalculator

class BarrierCache:
  """Persistent key-value store for NEB results, backed by a single SQLite file.

  Why SQLite instead of a dict / JSON / pickle file?
      - Survives restarts and crashes (results live on disk, not in memory).
      - Primary-key index: lookups stay microsecond-fast even with millions
        of entries (a CSV would need a full scan).
      - Atomic writes: a crash can never leave a half-written entry behind.
      - It is a real database: the cache doubles as a queryable dataset
        (coverage stats, provenance, future training data for a barrier
        surrogate).
      - Stdlib only: `import sqlite3`, no server, no config.

  Usage pattern: get(key) before running a NEB (cache hit = free),
  put(...) after (cache miss results are stored forever). Keys are the MD5
  fingerprints produced by MACENEBBarrierCalculator._env_key.
  """
  
  def __init__(self, db_path):
    # Open the database file; created automatically on first use.
    # Every SQL statement in this class flows through this connection.
    self.conn = sqlite3.connect(str(db_path))
    self.conn.execute("""
            CREATE TABLE IF NOT EXISTS barriers (
                -- MD5 fingerprint of (structures + model + NEB settings).
                -- PRIMARY KEY = unique (same environment can't be stored
                -- twice) and automatically indexed (fast lookups).
                key TEXT PRIMARY KEY,
                -- Forward barrier in eV (REAL = SQLite's float type).
                barrier REAL,
                -- Did FIRE converge? SQLite has no bool type: stored 0/1.
                converged INTEGER,
                -- Atoms in the NEB band (useful for coverage/timing stats).
                n_atoms INTEGER,
                -- Which model produced this result; prevents silently
                -- mixing barriers from different model versions.
                model_id TEXT,
                -- NEB protocol (cluster radii, n_images, fmax) as JSON
                -- text, so entries stay interpretable outside this code.
                settings TEXT,
                -- Full energy profile (eV relative to image 0) as JSON text.
                profile TEXT,
                -- Free provenance: SQLite fills the UTC timestamp itself.
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""))
            
    # SQLite works in transactions: changes are provisional until
    # commit(), which flushes them durably to disk.
    self.conn.commit()
    
  def get(self, key):
    """Return the stored result for `key` as a dict, or None on miss.

    The `?` is a parameter placeholder: the value travels separately in
    the tuple, so SQLite handles escaping (no string-concatenation bugs).
    fetchone() returns the row as a tuple in SELECT order —
    (barrier, converged, profile) — or None if the key is absent.
    """
    row = self.conn.execute(
        "SELECT barrier, converged, profile FROM barriers WHERE key=?",
        (key,)).fetchone()
    if row is None:
      return None
    # Rebuild Python types from the SQL row: float, bool, and the JSON
    # text parsed back into the list of per-image energies.
    return {"barrier":row[0], "converged": bool(row[1]), 
            "profile": json.loads(row[2])}
            
  def put(self, key, barrier, converged, n_atoms, model_id, settings, profile):
    """Store (or overwrite) one NEB result.

    INSERT OR REPLACE = upsert: if this key exists (environment was
    recomputed), the row is updated instead of raising a uniqueness
    error. The seven `?` placeholders are filled positionally by the
    tuple, mirroring the conversions of get(): bool -> int, dict/list
    -> JSON text. The 8th column (created_at) is filled by SQL itself.
    """
    self.conn.execute(
      "INSERT OR REPLACE INTO barriers VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
      (key, barrier, int(converged), n_atoms, model_id,
      json.dumps(settings), json.dumps(profile)))
    self.conn.commit()
    

class MACENEBBarrierCalculator:
  """Loads a MACE model once, builds NEB bands (periodic or or frozen-shell
    cluster), computes migration barriers with CI-NEB, and caches every
    result in SQLite so no environment is ever computed twice."""
  
  def __init__(self, model_source, cache_dir="data/cache/neb_cache", device="cuda",
               default_dtype="float64", n_images=5, fmax=0.05, max_steps=300,
               cluster=None, model_filename="model.model"):     
    """
        model_source : local path to a .model file, or Hugging Face repo ID
        n_images     : total NEB images, including the two endpoints
        fmax/max_steps: FIRE convergence criterion and step budget
        cluster      : None -> full periodic NEB (reference/validation mode)
                       dict(R_active=5.0, R_shell=7.0) -> spherical cluster:
                       atoms beyond R_shell discarded; atoms in
                       (R_active, R_shell] kept but frozen (passivate the cut)
    """
    # --- NEB protocol settings -------------------------------------
    self.device, self.dtype = device, default_dtype
    self.n_images, self.fmax, self.max_steps = n_images, fmax, max_steps
    self.cluster = cluster
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Resolve the model ---------------------------------------
    # Local file wins; anything else is treated as a HF repo ID and
    # downloaded once (hf_hub_download keeps its own disk cache).
    p = Path(model_source)
    if p.exists():
      self.model_path = p
    else:
      from huggingface_hub import hf_hub_download
      self.model_path = Path(hf_hub_download(
        repo_id=model_source, filename=model_filename,
        cache_dir=cache_dir / "hf"))
    self.model_id = self.model_path.name
    
    # --- 2. Barrier cache ---
    self.cache = BarrierCache (cache_dir/ "barrier.db") 
  
  def _new_image_calculator(self):
    """One MACE calculator per image: an ASE calculator binds to a single
    Atoms object, and NEB needs forces on all images simultaneously.
    Import is lazy so torch/mace stay optional for the rest of Kinetix."""
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=[str(self.model_path)],
                          device=self.device, default_dtype=self.dtype)
                          
  def prepare_band(self, start, end, migrating_index=None):
    """Build the list of NEB images (the 'band').

    Returns (images, n_frozen). Every image has the same atoms in the
    same order, a NEB requirement.
    """
    assert len(start) == len(end), "Start and end must have identical atoms"
    
    # --- 1. Locate the hop ------------------------------------------
    # Migrating atom = largest IS->FS displacement (overridable).
    if migrating_index is None:
      disp = np.linalg.norm(end.positions - start.positions, axis=1)
      migrating_index = int(np.argmax(disp))
    
    # Sphere center = hop midpoint, so origin and destination are
    # covered symmetrically by the cluster cut.
    center = 0.5 * (start.positions[migrating_index]
                    + end.positions[migrating_index])
    
    # --- 2. Cut the cluster (or skip in reference mode) --------------
    if self.cluster is None:
      # Full periodic cell, nothing frozen: the validation reference
      band_start, band_end, frozen = start, end, np.array([], int)
    else:
      r_a = self.cluster['R_active'] # inner region: free to relax
      r_s = self.cluster['R_shell'] # outer shell : kept but frozen
      
      # Distance of every atom to the center, minimum-image aware.
      d, _ = get_distances(start.positions, center[None, :],
                           cell=start.cell, pbc=start.pbc.any())
      
      d = d[:,0]
      mask = d <= r_s # atoms kept in the cluster
      shell = (d > r_a) & mask # kept atoms forming the frozen shell
      # FixAtoms indices must refer to the REDUCED cluster, not the
      # original cell: shell[mask] re-expresses shell flags in subset
      # order before nonzero() does the remapping.
      frozen = np.nonzero(shell[mask])[0] # atoms to be frozen during relaxation
      
      # Same boolean mask on both endpoints -> identical atoms/order.
      band_start, band_end = start[mask], end[mask]
      
    # --- 3. Assemble the band -----------------------------------------
    # Endpoints exact; intermediate images start as copies of the initial
    # state and are repositioned later by neb.interpolate().
    images = [band_start.copy() for _ in range(self.n_images)]
    images[-1] = band_end.copy()
    
    # --- 4. Attach calculators and constraints -------------------------
    for im in images:
      im.calc = self._new_image_calculator()
      if len(frozen):
        # The shell stays put in EVERY image while FIRE relaxes the
        # band; it mimics the stiffness of the removed bulk.
        im.set_constraint(FixAtoms(indices=frozen)) # every image gets the same frozen set (shell) while FIRE relaxes the band
    return images, len(frozen)
    
  def _env_key(self, start, end):
    """Fingerprint of the whole computation (structures + model + NEB
    settings) -> SQLite primary key.

    Identical physics (after 1 pm rounding) -> identical bytes -> cache
    hit; any real difference -> (almost certainly) a different key.
    """
    h = hashlib.md5()
    for at in (start, end):
     h.update(at.numbers.tobytes()) # composition/order
     h.update(np.round(at.positions, 3).tobytes()) # geometry, 1 pm bins
     h.update(np.round(at.cell[:], 3).tobytes()) # periodic context
    h.update(self.model_id.encode())  # which model
    h.update(json.dumps([self.cluster, self.n_images, self.fmax]).encode()) # NEB protocol
    return h.hexdigest()
    
  def compute_barrier(self, start, end, migrating_index=None, 
                      use_cache=True, full_output=False):
    """Barrier for the hop start->end: cache first, CI-NEB on miss."""
    # --- 1. Cache lookup ---------------------------------------------
    key = self._env_key(start, end)
    if use_cache:
      hit = self.cache.get(key)
      if hit is not None:
        return hit if full_output else hit["barrier"]
    
    # --- 2. Run CI-NEB -------------------------------------------------
    images, _ = self.prepare_band(start, end, migrating_index) 
    neb = NEB(images, climb=True) # climbing image converges on the TS
    neb.interpolate("idpp", mic=bool(start.pbc.any())) # seed the path
    
    t0 = time.time()
    opt = FIRE(neb, logfile=None)
    opt.run(fmax=self.fmax, steps=self.max_steps)
    try: 
      converged = bool(opt.converged())
    except Exception:
      converged = False
    wall = time.time() - t0 
    
    # --- 3. Extract the barrier -----------------------------------------
    # Energies relative to the initial image; barrier = highest point.
    rel = np.array([im.get_potential_energy() for im in images])
    rel = (rel - rel[0]).tolist()
    result = {"barrier": float(max(rel)), "converged": converged,
              "profile": rel, "n_atoms": len(images[0]), "wall_time": wall}
    
     # --- 4. Store forever ------------------------------------------------
    self.cache.put(key, result["barrier"], converged, len(images[0]),
                   self.model_id,
                   {"cluster": self.cluster, "n_images": self.n_images,
                    "fmax": self.fmax}, rel)
                    
    return result if full_output else result["barrier"]
    
  # ------------------------------------------------------------------ #
  def radius_scan(self, start, end, radii, shell=2.0):
    """Validation helper: barrier & wall time vs cluster radius.

    Bypasses the cache so every radius is genuinely recomputed, and
    restores the original cluster setting afterwards.
    """
    rows = []
    for r in radii:
      self.cluster = {"R_active": r, "R_shell": r + shell}
      res = self.compute_barrier(start, end, use_cache=False,
                                 full_output=True)
      rows.append((r, res["barrier"], res["wall_time"]))
    return rows
    

class KinetixMACEAdapter(ActivationEnergyCalculator):
    """Bridges Kinetix lattice hops to the ASE-native NEB calculator.

    Builds IS/FS clusters by SWAPPING the moving atom between the two hop
    endpoints (correct for both vacancies and interstitials).
    """
    
    DEFAULT_SPECIES_MAP = {   # Kinetix label -> element (None = no atom)
        "Hf": "Hf", "O": "O", "O_i": "O", "H": "H",
        "V_O": None, "Empty": None,
    }
    
    def __init__(self, model_source, species_map=None, cell=None, **neb_kwargs):
      self.species_map = dict(self.DEFAULT_SPECIES_MAP, **(species_map or {}))
      self.cell = cell
      self.neb = MACENEBBarrierCalculator(model_source, **neb_kwargs)
      
    def get_barrier(self, lattice, origin_idx, dest_idx, event_id=None):
      start, end = self.build_pair(lattice, origin_idx, dest_idx)
      return self.neb.compute_barrier(start, end)
      
    def build_pair(self, lattice, i0, i1):
      e0 = self._element(lattice[i0].chemical_specie)
      e1 = self._element(lattice[i1].chemical_specie)
      if (e0 is None) == (e1 is None):
        raise ValueError(f"Hop {i0}->{i1}: exactly one endpoint must "
                         f"host the moving atom (get {e0!r}, {e1!r})")
                         
      elem = e0 or e1
      p0 = np.asarray(lattice[i0].position, float)
      p1 = np.asarray(lattice[i1].position, float)
      cell = self._cell(lattice)
      radius = self.neb.cluster["R_shell"] if self.neb.cluster else np.inf
      center = 0.5 * (p0 + p1)
      
      pos = np.array([np.asarray(s.position, float) for s in lattice])
      
      d, _ = get_distances(pos, center[None, :], cell=cell, pbc=True)
      d = d[:, 0]
      
      bg_symbols, bg_positions = [], []
      for idx, site in enumerate(lattice):
        if idx in (i0, i1) or d[idx] > radius:
          continue
        el = self._element(site.chemical_specie)
        if el is None:
          continue
        bg_symbols.append(el)
        bg_positions.append(pos[idx])
        
      start_pos = p0 if e0 is not None else p1
      end_pos = p1 if e0 is not None else p0
      start = Atoms(symbols=bg_symbols + [elem],
                    positions=bg_positions + [start_pos], cell=cell, pbc=True)
      end = Atoms(symbols=bg_symbols + [elem],
                  positions=bg_positions + [end_pos], cell=cell, pbc=True)
      
      return start, end
      
    def _element(self, label):
      if label not in self.species_map:
        raise KeyError(f"Unknown species label {label!r}: extend species_map")
      return self.species_map[label]
      
    def _cell(self, lattice):
      if self.cell is not None:
        return np.asarray(self.cell, float)
      cell = getattr(lattice, "cell", None)
      if cell is not None:
        return np.asarray(cell, float)
      raise ValueError("No cell available: pass cell= to KinetixMACEAdapter")

    