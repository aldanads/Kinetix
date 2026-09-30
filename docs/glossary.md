# Kinetix Cross-Domain Glossary

> Terms bridge the three domains the project couples:
> **kinetic Monte Carlo (KMC)** materials simulation,
> **machine learning (ML)** barrier prediction, and **neuromorphic /
> memristive device** physics.
>
> Sources scanned: `README.md`, `AGENTS.md`, and package docstrings under
> `kinetix/` (notably `calculators/base.py`, `calculators/mace_neb.py`,
> `calculators/active_learning.py`, `utils/superbasin.py`,
> `lattice/kmc_loop.py`, `solvers/electrical.py`, `utils/metadata.py`).

| Term | Definition | Domain Context |
|---|---|---|
| Active learning (loop) | Iterative ML workflow: ML-predicted barriers that fail physical sanity checks are exported (with structures and metadata) to a DFT queue (`manifest.json` + one folder per barrier) for higher-fidelity recomputation, and the results refine the MACE model. In Kinetix the *export* stage exists. | ML ↔ KMC |
| Activation energy (barrier, $E_\mathrm{act}$) | Energy (eV) that must be overcome for an elementary event to occur; the central quantity linking KMC event rates to ML barrier prediction. Read from tabulated `data/parameters/activation_energies/` JSON in the production path; optionally predicted by a calculator (`ActivationEnergyCalculator.get_barrier`). | KMC ↔ ML |
| Arrhenius rate law | First-order transition rate $k_i = \nu_0 \exp(-E_{\mathrm{act},i}/k_\mathrm{B}T)$ assigned to each elementary event; the rate that feeds BKL selection and the kMC clock. | KMC |
| Attempt frequency ($\nu_0$) | Pre-exponential factor for event rates; $\nu_0 = 7\times10^{12}$ s⁻¹ (bond vibration) in Kinetix. | KMC |
| BKL algorithm (Bortz–Kalos–Lebowitz) | Rejection-free Monte Carlo selection: an event is drawn with probability $k_i/\sum_j k_j$ and always executed (no trial-and-error rejections). Kinetix implements it on a balanced binary tree of transition rates. | KMC |
| Balanced binary tree | Data structure holding all transition rates so BKL event selection and post-event rate updates run in $O(\log n)$ (`kinetix/utils/balanced_tree.py`); depends on the fixed `Event.catalog_tuple()` format. | KMC (implementation) |
| Barrier cache | Persistent SQLite key-value store of NEB results keyed by local structure *and* model file name (`model_id`), so restarts never recompute or silently mix barriers from different MACE model versions. | ML |
| CI-NEB (climbing-image nudged elastic band) | NEB variant in which the optimal image climbs to the saddle point, giving the minimum-energy barrier for a hop; the method behind the MACE NEB adapter (`kinetix/calculators/mace_neb.py`). | ML ↔ KMC |
| Conductive filament | Connected path of migrated defects/atoms bridging the electrodes; its formation and dissolution switches the device between resistance states — the physical object Kinetix's kMC simulates. | KMC ↔ Neuromorphic |
| Defect | Mobile point entity (vacancy, interstitial, redox-active species) carrying all dynamic state — chemical specie, charge, passivation level, enabled events — as a `Defect` object hosted by exactly one `Site`. | KMC |
| DFT (density functional theory) | First-principles reference method: source of the tabulated activation energies and the higher level of theory for recomputing flagged barriers in the active-learning queue. | ML ↔ KMC |
| Dirty site | A lattice site whose neighbourhood was affected by the executed event and whose rates/events must therefore be recomputed (bookkept by `EventHandler`). | KMC |
| ECM (electrochemical metallization) | Resistive-switching mechanism in which a metal filament (e.g., Ag) grows/dissolves via cation migration; one of the two device mechanisms Kinetix supports alongside VCM. | Neuromorphic ↔ KMC |
| Electrode scavenging | Electrodes acting as source/sink for mobile ions (e.g., oxygen scavenging), altering defect concentrations during switching. | KMC ↔ Device |
| Event | One catalogueable elementary process (migration, generation, redox, reaction) with origin, destination, barrier and rate; serialized via `Event.catalog_tuple()` — the fixed shape the balanced tree and superbasin depend on. | KMC |
| Event catalog | The full list of currently available events at a kMC step; its digest is recorded every step in the golden-trace fixtures as the physics-preservation contract. | KMC |
| Field-assisted barrier | Reduction of a charged species' activation energy by the local electric field, $E_\mathrm{act} - q\mathbf{E}\cdot\mathbf{d}$; couples the Poisson solution to defect kinetics. | KMC ↔ Multiphysics |
| FEM (finite element method) / DOLFINx | FEniCSx stack solving the Poisson and heat equations on gmsh meshes; the multiphysics layer of the kMC loop (Linux-only solver paths). | Multiphysics |
| Filament percolation | Criterion that a connected defect path spans the electrodes (`is_filament_percolating`); structurally marks the switched (low-resistance) state of the device. | KMC ↔ Neuromorphic |
| Forming | Initial soft-breakdown step that creates the first conductive path in a pristine device, after which SET/RESET cycling operates. Not referenced anywhere in the Kinetix codebase; field-standard meaning only. | Neuromorphic |
| Golden trace | Byte-level contract of recorded kMC steps (times, rates, event digests) in `tests/test_golden_trace.py` + fixtures that must remain unchanged across refactors — the project's physics-preservation guarantee. | KMC (verification) |
| Grain boundary (GB) | Crystallographic interface where migration barriers are modified with planar/cylindrical, including direction-dependent entry/exit/within-GB barriers. | KMC |
| HRS / LRS (high/low resistance state) | The distinct resistance levels of a memristor that encode stored information; HRS ↔ LRS transitions correspond to filament dissolution/formation. The acronyms are not used in the code; field-standard names for the states Kinetix produces. | Neuromorphic |
| I–V curve | Current-vs-voltage response of the device, produced by `ElectricalController.plot_V_I` / `save_IV_csv`; its hysteresis loop is the experimental fingerprint of resistive switching. | Neuromorphic ↔ Device |
| kMC (kinetic Monte Carlo) | Stochastic simulation method that advances *physical* time by selecting real events with probabilities proportional to their rates; Kinetix's core algorithm (rejection-free BKL on a balanced binary tree). | KMC (core) |
| Lattice site (`Site`) | Node of the 3D grid holding topology (neighbours, interface flags, sublattice/`site_type`) and exactly one `Defect`; the unit of simulation state. | KMC |
| MACE | Machine-learning interatomic potential (equivariant message-passing architecture) used in Kinetix to compute migration barriers through CI-NEB; models are fetched from the Hugging Face Hub. | ML |
| Materials Project (MP) | DFT database from which crystal structures (`mp-*` IDs) and summary data are fetched (with local JSON cache) to seed the lattice; supplies `crystal_system`/`space_group` to `metadata.json`. | Data / KMC |
| Memristor / ReRAM | Two-terminal device whose resistance depends on the history of applied voltage/current; the physical substrate of synaptic elements in neuromorphic hardware. Kinetix's `simulation_type: electronic_device` models these. | Neuromorphic ↔ KMC |
| Migration pathway | Allowed hop between neighbouring sites for a given species, with its own activation energy (including grain-boundary direction dependence). | KMC |
| MLIP (machine-learning interatomic potential) | ML model mapping atomic configurations to energies/forces; in Kinetix it is extended from geometry to *barrier prediction* via NEB. | ML |
| Multiphysics coupling | Real-time feedback between electrostatics (Poisson), Joule heating (heat equation) and stochastic defect kinetics within the kMC loop, re-solving fields at each voltage update. | Multiphysics ↔ KMC |
| NEB (nudged elastic band) | Method that finds the minimum-energy path and barrier between two known endpoints by relaxing a chain of images under spring forces; used with MACE to generate migration barriers. | ML ↔ KMC |
| Passivation | Deactivation of a defect by trapping (e.g., vacancy captured with H); the integer `passivation_level` keys the activation energies (`Act_E[str(level)]`) and gates capture/depassivation events. | KMC |
| Poisson equation | $-\nabla\cdot(\varepsilon_0\varepsilon_r\nabla V)=\rho$ for the electric potential from ionic charge density; switches to the current-continuity form $-\nabla\cdot(\sigma\nabla V)=0$ once a filament percolates. | Multiphysics |
| Provenance (metadata) | The `metadata.json` block recording creator, affiliation, DOI/repository identifiers, git commit/branch and Materials Project data — the FAIR reproducibility record of a run. | Software / FAIR |
| Rare event | Low-probability transition that dominates long-timescale kinetics; the reason kMC exists and the bottleneck superbasin acceleration escapes. | KMC |
| Redox reaction | Oxidation/reduction event changing a species' charge state at a site (e.g., metal cation deposition/dissolution); central to ECM switching. | KMC ↔ Neuromorphic |
| Resistive switching | Reversible formation/dissolution of a conductive path changing device resistance (VCM or ECM mechanisms); the basis of RRAM and of analog memory for neuromorphic computing. | Neuromorphic ↔ KMC |
| Superbasin | Set of states connected by fast internal transitions, escaped from as a single unit to bypass rare-event bottlenecks (local superbasin method after Fichthorn & Lin 2013, cited in `utils/superbasin.py`). | KMC |
| Synaptic weight | Analog weight of an artificial synapse, stored as device conductance and updated by programming pulses (potentiation/depression). Not implemented or referenced in Kinetix; contextual meaning for the neuromorphic application only. | Neuromorphic |
| Tabulated barriers (`Act_E_dict`) | Production barrier source: JSON activation-energy tables dispatched by `Site.transition_rates`. Note the ML bridge is currently *contract-only* — `CalculatorConfig.type` (`"mace_neb"` vs `"tabulated"`) is inert and no calculator is dispatched at runtime. | KMC ↔ ML |
| Uncertainty quantification (UQ) | Optional per-barrier prediction uncertainty exposed by `ActivationEnergyCalculator.uncertainty()` (`None` = not available); intended to drive active-learning triage. Not implemented by any calculator yet. | ML |
| VCM (valence-change mechanism) | Resistive-switching mechanism driven by migration and charge-state changes of anion vacancies (e.g., O vacancies in HfO₂); Kinetix's flagship scenario (`VCM_mock` preset). | Neuromorphic ↔ KMC |
| Voltage protocol | Applied voltage-vs-time schedule (`CONSTANT`, `RAMP_CYCLE`, `ZERO_HOLD`); `RAMP_CYCLE` applies a full 4-phase switching cycle (0→V_max→0→V_min→0) used to sweep I–V hysteresis. | Neuromorphic ↔ Device |
| Wulff shape | Equilibrium crystal shape construction used to analyse deposition morphology and surface relaxation. | KMC (deposition) |

*Legend: **KMC** = kinetic Monte Carlo / materials simulation, **ML** = machine
learning, **Neuromorphic** = memristive-device / neurocomputing context,
**Multiphysics** = FEM field solvers, **Data/Software/FAIR** = supporting
infrastructure.
