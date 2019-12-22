# AMD Zen (Dec 2019)

## Python

```
ssh jdh4@perseus-amd.princeton.edu
module load anaconda3/2019.10
OMP_NUM_THREADS=<num> python lu.py
```

| cluser      | OMP_NUM_THREADS | Execution time (s) |
|:-----------:|:---------------:|:------------------:|
| AMD Perseus | 1               | 
| AMD Perseus | 2               | 34.9 |
| AMD Perseus | 4               | 18.0 |
| AMD Perseus | 8               |  9.8 |
| AMD Perseus | 16              |  6.1 |
| AMD Perseus | 32              |  3.9 |

The execution times are the best of 5 runs.

Below is the Python script:

```
from time import perf_counter

import numpy as np
import scipy as sp
from scipy.linalg import lu

N = 10000
cpu_runs = 5

times = []
X = np.random.randn(N, N).astype(np.float64)
for _ in range(cpu_runs):
  t0 = perf_counter()
  p, l, u = lu(X, check_finite=False)
  times.append(perf_counter() - t0)
print("CPU time: ", min(times))
print("NumPy version: ", np.__version__)
print("SciPy version: ", sp.__version__)
print(p.sum())
print(times)
```
