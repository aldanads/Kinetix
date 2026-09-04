# kinetix/calculators/mace_neb.py
"""mace_neb.py � MACE CI-NEB barrier calculator with model & barrier caching."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from itertools import product
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import get_distances
from ase.mep import NEB
from ase.optimize import FIRE
from ase.data import chemical_symbols


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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
            
    # SQLite works in transactions: changes are provisional until
    # commit(), which flushes them durably to disk.
    self.conn.commit()
    
  def get(self, key):
    """Return the stored result for `key` as a dict, or None on miss.

    The `?` is a parameter placeholder: the value travels separately in
    the tuple, so SQLite handles escaping (no string-concatenation bugs).
    fetchone() returns the row as a tuple in SELECT order �
    (barrier, converged, profile) � or None if the key is absent.
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
                          
  def prepare_band(self, start, end, migrating_index=None, frozen=None):
    """Build the list of NEB images (the 'band').

    Returns (images, n_frozen). Every image has the same atoms in the
    same order, a NEB requirement.
    
    frozen : precomputed shell indices (adapter mode). When provided,
             start/end are assumed to be already-cut clusters and the
             distance-based derivation below is skipped entirely.
    """
    assert len(start) == len(end), "Start and end must have identical atoms"
    
    if frozen is not None:
      # --- Adapter mode: cluster already cut, shell already classified.
      band_start, band_end = start, end
      frozen = np.asarray(frozen, int)
      
      
    elif self.cluster is None:
      # Full periodic cell, nothing frozen: the validation reference
      band_start, band_end, frozen = start, end, np.array([], int)
      
      
    else:
      # --- Standalone cluster mode: derive everything from the structures.
        # Migrating atom = largest IS->FS displacement (overridable).
      if migrating_index is None:
        disp = np.linalg.norm(end.positions - start.positions, axis=1)
        migrating_index = int(np.argmax(disp))
      
      # Sphere center = hop midpoint, so origin and destination are
      # covered symmetrically by the cluster cut.
      center = 0.5 * (start.positions[migrating_index]
                      + end.positions[migrating_index])
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
    
  def compute_barrier(self, start, end, migrating_index=None, frozen=None, 
                      use_cache=True, full_output=False):
    """Barrier for the hop start->end: cache first, CI-NEB on miss."""
    # --- 1. Cache lookup ---------------------------------------------
    key = self._env_key(start, end)
    if use_cache:
      hit = self.cache.get(key)
      if hit is not None:
        return hit if full_output else hit["barrier"]
    
    # --- 2. Run CI-NEB -------------------------------------------------
    images, _ = self.prepare_band(start, end, migrating_index, frozen) 
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
    original = self.cluster
    try:
      for r in radii:
        self.cluster = {"R_active": r, "R_shell": r + shell}
        res = self.compute_barrier(start, end, use_cache=False,
                                   full_output=True)
        rows.append((r, res["barrier"], res["wall_time"]))
    finally:
      self.cluster = original
    return rows
    

class KinetixMACEAdapter(ActivationEnergyCalculator):
    """Bridges Kinetix lattice hops to the ASE-native NEB calculator."""
    
    def __init__(self, model_source, kx, species_map=None, pbc=(True, True, False), **neb_kwargs):
      self.kx = kx # Provides _kdtree, MIC, configs
      self.pbc = pbc # Periodic in plane
      self.species_map = species_map or self.build_species_map(kx)
      self.cell = np.array(kx.structure.lattice.matrix, float) # Angstroms
      self.neb = MACENEBBarrierCalculator(model_source, **neb_kwargs)
      
      
      # The adapter only makes sense in cluster mode: a kMC supercell as
      # one big Atoms object would put thousands of atoms in a NEB.
      # Fail at construction (startup), not mid-simulation.
      if self.neb.cluster is None:
            raise ValueError(
                "KinetixMACEAdapter requires cluster={'R_active': ..., "
                "'R_shell': ...} in the calculator options; use "
                "MACENEBBarrierCalculator directly for periodic reference runs")
      
    def get_barrier(self, lattice, origin_idx, dest_idx, event_id=None,
                    full_output=False):
      start, end, frozen = self.build_pair(lattice, origin_idx, dest_idx)
      return self.neb.compute_barrier(start, end, frozen=frozen,
                                      full_output=full_output)
      
    def build_species_map(self,kx, extra_species=None):
      """
      Kinetix label -> element (None = no atom).

      Host elements: pristine structure composition (Hf, O, ...).
      Defect/host-like labels: defects_config via `physical_element`
      (null = pseudo-particle). Passivated levels map to `passivant`.
      Missing semantics raise instead of guessing material physics.
      """
      smap = dict(extra_species or {})
      smap['Empty'] = None
      
      # --- host lattice, from the pristine structure ---------------------
      comp = getattr(getattr(kx, "structure_basic", None), "composition", None)
      if comp is not None:
        for elm in comp.as_dict():
          smap[str(elm)] = str(elm)
          
      # --- defects and host-like entries (lattice_oxygen) -----------------
      for name, cfg in (getattr(kx, "defects_config", None) or {}).items():
        sym = cfg.get("symbol")
        if "physical_element" in cfg:
          smap[sym] = cfg["physical_element"]
        elif sym in chemical_symbols:
          smap[sym] = sym
        else:
          raise KeyError(
            f"defect '{name}': add 'physical_element' to its config "
            f"('O' for O_i, null for V_O)")
        n = int(cfg.get("max_passivation_level") or 0)
        if n > 0:
          pas = cfg.get("passivant")
          if not pas:
            raise KeyError(f"defect '{name}': max_passivation_level>0 "
                           f"requires a 'passivant' field")
          
          for lvl in range(1, n+1):
            smap[f"{sym}_{lvl}"] = pas
          
      return smap
    
    # -- species resolution ------------------------------------------------  
    def _site_elements(self, site):
      """Elements contributed by the site (list; empty = no atom) """
      label = site.chemical_specie
      if label not in self.species_map:
        raise KeyError(f"Unknown species {label!r}: add 'element' to "
                       f"defects_config or host composition")
      els = [self.species_map[label]] if self.species_map[label] else []
      lvl = getattr(site, "passivation_level", 0)
      if lvl: # V_O + nH -> n H atoms
        dname = site._get_current_defect_name()
        els += [site.defects_config[dname]["passivant"]] * lvl
      return els
      
    # -- sphere via your KD-tree --------------------------------------------
    def _candidate_keys(self, center):
      r = self.neb.cluster["R_shell"] if self.neb.cluster else 8.0
      per_shifts = []
      for vec, per, in zip(self.cell, self.pbc):
        per_shifts.append([np.zeros(3), vec, -vec] if per else [np.zeros(3)])
      found=set()
      for s in (np.sum(c, axis=0) for c in product(*per_shifts)):
        found.update(self.kx._kdtree.query_ball_point(center + s, r))
      return [self.kx._kdtree_indices[i] for i in found]
      
    def build_pair(self, grid, o, d):
      els_o, els_d = self._site_elements(grid[o]), self._site_elements(grid[d])

      if not (bool(els_o) ^ bool(els_d)):
            raise ValueError(f"Hop {o}->{d}: exactly one endpoint must host "
                             f"the moving atom")
                         
      p0 = np.asarray(grid[o].position, float)
      p1 = np.asarray(grid[d].position, float)
      center = 0.5 * (p0 + p1)
      r_a = self.neb.cluster["R_active"]
      r_s = self.neb.cluster["R_shell"]
      
      symbols, positions, frozen = [], [], []
      for k in sorted(self._candidate_keys(center)): # sorted = deterministic
        if k in (o, d):  # atom order -> stable cache keys
          continue
        site = grid[k]
        els = self._site_elements(site)  
        if not els: # Empty / bare vacancy
          continue
        v = self.kx._minimum_image_vector(np.array(site.position, float) - center)
        dm = np.linalg.norm(v)
        if dm > r_s: # outside the cluster
          continue
        is_shell = dm > r_a # Kept, but frozen
        for i, el in enumerate(els):
          if is_shell:
            frozen.append(len(symbols))
          symbols.append(el)
          positions.append(center + v + i * 0.05 * np.array([1., 0., 0.]))
        
      start_pos = p0 if els_o else p1
      end_pos = p1 if els_o else p0
      moving = els_o or els_d
      if len(moving) != 1:
          raise NotImplementedError(
              f"Hop {o}->{d}: multi-atom migrating species not supported yet")
      start = Atoms(symbols=symbols + moving,
                    positions=positions + [start_pos], cell=self.cell, pbc=self.pbc)
      end = Atoms(symbols=symbols + moving,
                  positions=positions + [end_pos], cell=self.cell, pbc=self.pbc)
      # frozen indices refer to background atoms only; the moving atom (last)
      # is never frozen � consistent in both images
      return start, end, frozen