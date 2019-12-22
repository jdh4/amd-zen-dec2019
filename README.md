# AMD Zen (Dec 2019)

## Python

```
ssh jdh4@perseus-amd.princeton.edu
module load anaconda3/2019.10
OMP_NUM_THREADS=<num> python lu.py
```

| cluser      | OMP_NUM_THREADS | Execution time (s) |
|:-----------:|:---------------:|:------------------:|
| AMD Perseus | 1               | 67.0 |
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

## GROMACS

```
wget ftp://ftp.gromacs.org/pub/benchmarks/rnase_bench_systems.tar.gz
tar -zxvf rnase_bench_systems.tar.gz
BCH=rnase_cubic
gmx grompp -f $BCH/pme_verlet.mdp -c $BCH/conf.gro -p $BCH/topol.top -o bench.tpr
gmx mdrun -ntomp 8 -s bench.tpr
gmx mdrun -ntmpi 16 -ntomp 1 -s bench.tpr
gmx mdrun -ntmpi 1 -ntomp 16 -s bench.tpr
```

| cluser      | MPI threads | OMP_NUM_THREADS | Execution time (s) |
|:-----------:|:-----------:|:---------------:|:------------------:|
| AMD Perseus | 1           | 1               | 67.0 |
| AMD Perseus | 1           | 2               | 34.9 |

GROMACS was built according to this procedure:

```bash
#!/bin/bash
version=2019.4
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version}.tar.gz
tar -zxvf gromacs-${version}.tar.gz
cd gromacs-${version}
mkdir build_stage1
cd build_stage1

#############################################################
# build gmx (stage 1)
#############################################################

module purge
module load aocc
module load rh/devtoolset/7

OPTFLAGS="-Ofast -DNDEBUG"

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DGMX_BUILD_OWN_FFTW=ON \
-DCMAKE_C_COMPILER=clang -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=OFF -DGMX_MPI=OFF -DGMX_OPENMP=ON \
-DGMX_DOUBLE=OFF \
-DGMX_GPU=OFF \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_COOL_QUOTES=OFF -DREGRESSIONTEST_DOWNLOAD=ON

make
make check
make install
```
