from initialization import initialization
from kinetix.calculators import KinetixMACEAdapter
from ase.io import write
import time
import numpy as np

def get_parameters_from_sim_id(sim_id: int) -> dict:
   """Map SIM_ID to simulation parameters."""
   # Define parameters
   v0_initial_concentrations = [1.0e-2, 2.0e-2, 3.0e-2,  4.0e-2,  5.0e-2]
   temperatures = [293.0, 310.0, 373.0, 473.0, 573.0]
   h_generation = [0.45,0.48,0.50, 0.52, 0.55]
   
   idx = sim_id
   
   i_vo = idx % 5
   i_temp = (idx // 5) % 5
   i_gen_h = (idx // 25) % 5
   
   return {
     'vo_initial_concentration': v0_initial_concentrations[i_vo],
     'temperature': temperatures[i_temp],
     'h_generation': h_generation[i_gen_h]
   }

# Pristine equivalence: symmetry-equivalent hops must agree
def find_bulk_equivalent_hops(grid, max_hops=5):
    """Find a few pristine interstitial->interstitial hops for equivalence testing."""
    hops = []
    for o, site in grid.items():
        if site.site_type != "interstitial" or site.chemical_specie != "Empty":
            continue
        if not (r_s <= site.position[2] <= Lz - r_s):
            continue
        for d in site.nearest_neighbors_idx:
            dest = grid[d]
            if dest.site_type == "interstitial" and dest.chemical_specie == "Empty":
                hops.append((o, d))
                if len(hops) >= max_hops:
                    return hops
    return hops

config_name = "VCM_mock.yaml"
sim_id = 0

# MACE model parameters
calculator_config = {
    "type": "mace_neb",
    "model": "HfO2_mh1_F_LONG_cpu.model",
    "options": {
        "device": "cpu",
        "cluster": {"R_active": 5.0, "R_shell": 7.0},
        "n_images": 5,
        "fmax": 0.05 
    }
}

r_s = calculator_config["options"]["cluster"]["R_shell"]

params = get_parameters_from_sim_id(sim_id)
System_state,_,_,_,_,_ = initialization(sim_id, params, config_name)

Lz = System_state.crystal.size[2]
center = np.array(System_state.crystal.size) / 2.0
best, origin_idx = np.inf, None
for idx, site in System_state.grid_crystal.items():
    if site.site_type != "interstitial" or site.chemical_specie != "Empty":
        continue
    if not (r_s <= site.position[2] <= Lz - r_s):
        continue
    d = np.linalg.norm(np.array(site.position) - center)
    if d < best:
        best, origin_idx = d, idx

if origin_idx is None:
    raise ValueError("No bulk-like empty interstitial found.")

cfg = System_state.defects_config["oxygen_interstitial"]
support_update_sites = set()
event_update_sites = set()
System_state._introduce_specie_site(origin_idx, support_update_sites, event_update_sites, cfg["symbol"], cfg["charge"])

dest_idx = next(n for n in System_state.grid_crystal[origin_idx].nearest_neighbors_idx
                if System_state.grid_crystal[n].site_type == "interstitial"
                and System_state.grid_crystal[n].chemical_specie == "Empty")

System_state.update_sites(support_update_sites, event_update_sites)


adapter = KinetixMACEAdapter(calculator_config["model"], kx=System_state, **calculator_config["options"])
grid = System_state.grid_crystal

c = 0.5 * (np.array(grid[origin_idx].position) + np.array(grid[dest_idx].position))
expected = {k for k,s in grid.items()
            if np.linalg.norm(System_state._minimum_image_vector(np.array(s.position) - c)) <= r_s}

assert expected == set(adapter._candidate_keys(c)), "candidate search mismatch"

start, end, frozen = adapter.build_pair(grid, origin_idx, dest_idx)
assert len(start) == len(end)
assert sorted(set(start.symbols)) == ["Hf", "O"]
assert np.allclose(start.positions[-1], grid[origin_idx].position)
assert np.allclose(end.positions[-1], grid[dest_idx].position)

write("mace_test_IS.extxyz", start)
write("mace_test_FS.extxyz", end)

ea = adapter.get_barrier(grid, origin_idx, dest_idx)
print(f'O_i hop barrier: {ea:.3f} eV')
t0 = time.perf_counter()
adapter.get_barrier(grid, origin_idx, dest_idx)
print(f"Cache hit time: {time.perf_counter() - t0:.4f} s")

equiv_hops = find_bulk_equivalent_hops(grid, max_hops=5)
bars = [adapter.get_barrier(grid, o, d) for o, d in equiv_hops]
print(f"Equivalence std: {np.std(bars):.4f} eV (expect ~0)")
assert np.std(bars) < 0.01, f"Equivalence violated: {bars}"