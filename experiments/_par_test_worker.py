import sys, os, time
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
import schwinger_core as core
import qiskit_backend as qb
from scipy.optimize import minimize

seed = int(sys.argv[1])
N,F,L = 3,2,2
ev = qb.NoisyEvaluator(N,F,L,ansatz='gi',fold_lambda=1,p=0.01,eps=core.SPAM_DEFAULT)
cost = ev.cost_fn(0.0)
rng = np.random.default_rng(seed)
theta0 = rng.uniform(-np.pi,np.pi,ev.n_params)
t0=time.time()
res = minimize(cost, theta0, method='COBYLA', options={'maxiter':2000,'rhobeg':0.5})
dt = time.time()-t0
print(f"worker seed={seed}: {dt:.1f}s E={res.fun:.4f}")
