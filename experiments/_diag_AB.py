"""Free diagnostics: (A) ISA-circuit diff job1 vs Job0, (B) per-group energies from S1/S2/job1 counts."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from qiskit.circuit import Delay
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from qiskit_ibm_runtime import QiskitRuntimeService
from ibm_taskB_step4_submit import build_groups

PINNED = [8, 9, 10, 11, 18, 31]
svc = QiskitRuntimeService()
backend = svc.backend("ibm_marrakesh")
target = backend.target
dt = target.dt

j1 = svc.job("daoeuotr85ps73ff6a70")
j0 = svc.job("daof9vdr85ps73ff6u10")
s_circs = [p[0] for p in j1.inputs["pubs"]]
e_circ = j0.inputs["pubs"][0][0]


def stats(qc, name):
    ops = qc.count_ops()
    n_cz = ops.get("cz", 0)
    skip = {"cz", "measure", "barrier", "delay"}
    n_1q = sum(v for k, v in ops.items() if k not in skip)
    pm = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)])
    sched = pm.run(qc)
    # per-qubit timeline
    per_q = {q: [] for q in PINNED}
    total = 0
    busy = {q: 0 for q in PINNED}
    idle_total = {q: 0 for q in PINNED}
    for ins in sched.data:
        qs = [sched.find_bit(x).index for x in ins.qubits]
        nm = ins.operation.name
        if nm == "delay":
            d = ins.operation.duration if ins.operation.unit == "dt" else ins.operation.duration / dt
        elif nm == "barrier":
            d = 0
        else:
            props = target[nm].get(tuple(qs))
            d = (props.duration / dt) if (props is not None and props.duration) else 0
        for q in qs:
            if q in per_q:
                per_q[q].append((ins.operation.name, d))
                if ins.operation.name == "delay":
                    idle_total[q] += d
                else:
                    busy[q] += d
    dur_dt = max(sum(d for _, d in per_q[q]) for q in PINNED)
    internal = {}
    for q in PINNED:
        seq = per_q[q]
        idx = [i for i, (n, _) in enumerate(seq) if n not in ("delay", "barrier")]
        if idx:
            a, b = idx[0], idx[-1]
            internal[q] = sum(d for n, d in seq[a:b + 1] if n == "delay")
        else:
            internal[q] = 0
    print(f"{name}: CZ={n_cz}  1q gates={n_1q}  depth={qc.depth()}  duration={dur_dt*dt*1e6:.2f} us"
          f"  | qubit31: total idle (incl. leading/trailing)={idle_total[31]*dt*1e6:.2f} us,"
          f" internal idle (between first/last op)={internal[31]*dt*1e6:.2f} us")
    print("      internal idle per qubit (us): " + ", ".join(f"{q}:{internal[q]*dt*1e6:.2f}" for q in PINNED))
    return dict(cz=n_cz, n1q=n_1q, depth=qc.depth())


print("=== A. ISA circuit diff ===")
for i, c in enumerate(s_circs):
    stats(c, f"job1 basis-circuit {i}")
stats(e_circ, "Job0 bare circuit    ")

print("\n=== B. per-group energy contributions (hardware counts) ===")
groups = build_groups()


def group_energies(counts_by_group):
    out = []
    for g, counts in zip(groups, counts_by_group):
        n = sum(counts.values())
        tot = 0.0
        for lbl, c in zip(g.paulis.to_labels(), g.coeffs):
            c = float(np.real(c))
            ex = 0.0
            for b, cnt in counts.items():
                s = 1
                for i, ch in enumerate(lbl):
                    if ch != "I" and b[i] == "1":
                        s = -s
                ex += s * cnt
            tot += c * ex / n
        out.append(tot)
    return out


rows = {}
for name, jid in [("job1", "daoeuotr85ps73ff6a70"), ("S1", "daofnv78gn2s739nqjlg"), ("S2", "daofnvopqrnc7399u9ig")]:
    r = svc.job(jid).result()
    cs = [getattr(r[i].data, list(r[i].data.keys())[0]).get_counts() for i in range(len(r))]
    rows[name] = group_energies(cs)
print("group labels:")
for gi, g in enumerate(groups):
    print(f"  g{gi}: {list(g.paulis.to_labels())}")
print(f"{'':6s}" + "".join(f"{'g'+str(i):>10s}" for i in range(5)) + f"{'total':>10s}")
for name, v in rows.items():
    print(f"{name:6s}" + "".join(f"{x:10.4f}" for x in v) + f"{sum(v):10.4f}")
