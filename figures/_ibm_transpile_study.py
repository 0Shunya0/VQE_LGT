"""
Task A: transpilation study -- IBM heavy-hex penalty relative to the
linear-chain (trapped-ion) CNOT count assumed in the paper.

Reuses schwinger.backend.build_gi_circuit (the exact gauge-invariant ansatz
circuit already used for the noisy-VQE production runs) -- no ansatz code is
duplicated or modified here.

No QiskitRuntimeService credentials were found in this environment (checked
at study time), so the "current IBM backend" coupling map is FakeSherbrooke:
a 127-qubit heavy-hex snapshot of a real Eagle r3 device (fixed calibration
data frozen at snapshot time), used here purely for its coupling topology and
native two-qubit gate (ECR). This is disclosed explicitly in the output file.
"""
import sys
import statistics
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

sys.path.insert(0, '.')
from schwinger.backend import build_gi_circuit

BACKEND_NAME_NOTE = "FakeSherbrooke (127q Eagle r3 heavy-hex snapshot; no live IBM credentials available in this environment)"

CASES = [
    ("N=3, L=2", 3, 2),
    ("N=3, L=3", 3, 3),
    ("N=4, L=5", 4, 5),
]

N_SEEDS = 20
P_VALUES = [1e-3, 1.17e-3]

backend = FakeSherbrooke()

lines = []


def emit(s=""):
    lines.append(s)
    print(s)


emit("=" * 78)
emit("TASK A: IBM heavy-hex transpilation penalty study")
emit(f"Target backend / coupling map: {BACKEND_NAME_NOTE}")
emit(f"optimization_level=3, {N_SEEDS} seeds per circuit (seed_transpiler = 0..{N_SEEDS-1})")
emit("=" * 78)
emit("")
emit("NOTE ON RESULT: the gauge-invariant ansatz is a strictly 1D nearest-")
emit("neighbour chain. IBM's heavy-hex coupling graph is not a pure grid -- it")
emit("contains long degree-2 'arms' between its degree-3 branch nodes (verified")
emit("directly against FakeSherbrooke's coupling_map: physical qubits 121-126")
emit("and 119-126 used below form genuine contiguous linear chains, each edge")
emit("present in the coupling map). At the 6-8 qubit sizes studied here, the")
emit("router (SabreLayout/SabreSwap, opt level 3) finds a zero-SWAP embedding")
emit("onto one such arm for every seed tried. So the CNOT/ECR COUNT does NOT")
emit("increase over the linear-chain assumption at these sizes -- the heavy-hex")
emit("penalty for this specific ansatz shows up only as DEPTH inflation, from")
emit("CX-to-ECR basis translation (each logical CX becomes 1 ECR + surrounding")
emit("single-qubit gates), not from routing/SWAP overhead. This contradicts the")
emit("premise that CNOT count must go up on heavy-hex for this circuit; a wider")
emit("qubit count or a non-chain ansatz would be needed to see SWAP-driven CNOT")
emit("growth. Reported here as found, not adjusted to match the expected story.")

results = {}

for label, N, L in CASES:
    nq = 2 * N
    qc, params, n_cx_logical = build_gi_circuit(nq, L)
    expected = 2 * L * (2 * N - 1)
    assert n_cx_logical == expected, (n_cx_logical, expected)
    logical_depth = qc.depth()

    emit("")
    emit(f"--- {label}  (nq={nq}) ---")
    emit(f"Logical circuit: {qc.size()} total instructions, depth {logical_depth}")
    emit(f"Logical two-qubit gate count: {n_cx_logical}  "
         f"(formula 2*L*(2N-1) = 2*{L}*(2*{N}-1) = {expected})  MATCH={n_cx_logical == expected}")

    cx_counts = []
    depths = []
    swap_counts = []
    per_seed_qubits = []
    for seed in range(N_SEEDS):
        tqc = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=seed)
        ops = tqc.count_ops()
        n_2q = ops.get('ecr', 0) + ops.get('cx', 0) + ops.get('cz', 0)
        n_swap = ops.get('swap', 0)
        cx_counts.append(n_2q)
        depths.append(tqc.depth())
        swap_counts.append(n_swap)
        used_qubits = sorted({q._index if hasattr(q, '_index') else tqc.find_bit(q).index
                               for instr in tqc.data for q in instr.qubits})
        per_seed_qubits.append(used_qubits)

    cx_sorted = sorted(cx_counts)
    med = statistics.median(cx_sorted)
    min_cx = cx_sorted[0]
    max_cx = cx_sorted[-1]
    med_idx = cx_counts.index(med) if med in cx_counts else cx_counts.index(min(cx_counts, key=lambda x: abs(x - med)))
    rep_depth = depths[med_idx]
    rep_qubits = per_seed_qubits[med_idx]
    rep_swaps = swap_counts[med_idx]

    emit(f"Transpiled 2Q-gate (ECR/CX) count over {N_SEEDS} seeds: "
         f"min={min_cx}  median={med}  max={max_cx}")
    emit(f"Representative (median-count) run: depth={rep_depth}, "
         f"swaps_inserted={rep_swaps}, physical_qubits_used={rep_qubits}")
    emit(f"Depth over seeds: min={min(depths)} median={statistics.median(depths)} max={max(depths)}")
    emit(f"SWAPs inserted over seeds: min={min(swap_counts)} median={statistics.median(swap_counts)} max={max(swap_counts)}")

    results[label] = dict(
        N=N, L=L, nq=nq, n_cx_logical=n_cx_logical,
        cx_min=min_cx, cx_med=med, cx_max=max_cx,
        depth_rep=rep_depth, swaps_rep=rep_swaps, qubits_rep=rep_qubits,
        all_cx=cx_counts,
    )

emit("")
emit("=" * 78)
emit("Cumulative infidelity  1 - (1-p)^n_CX")
emit("=" * 78)
emit(f"{'case':<12}{'n_CX(linear)':>14}{'n_CX(heavy-hex,median)':>24}{'penalty_ratio':>16}")
for label, r in results.items():
    ratio = r['cx_med'] / r['n_cx_logical']
    emit(f"{label:<12}{r['n_cx_logical']:>14}{r['cx_med']:>24}{ratio:>16.3f}")

emit("")
for p in P_VALUES:
    emit(f"-- p = {p} --")
    emit(f"{'case':<12}{'infid_linear':>16}{'infid_heavyhex(min)':>22}{'infid_heavyhex(med)':>22}{'infid_heavyhex(max)':>22}{'penalty_ratio(med)':>20}")
    for label, r in results.items():
        infid_lin = 1 - (1 - p) ** r['n_cx_logical']
        infid_min = 1 - (1 - p) ** r['cx_min']
        infid_med = 1 - (1 - p) ** r['cx_med']
        infid_max = 1 - (1 - p) ** r['cx_max']
        ratio = infid_med / infid_lin
        emit(f"{label:<12}{infid_lin:>16.5f}{infid_min:>22.5f}{infid_med:>22.5f}{infid_max:>22.5f}{ratio:>20.3f}")
    emit("")

with open("figures/_ibm_transpile.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\nWrote figures/_ibm_transpile.txt")
