# AMD Zen (Dec 2019)

## Python

```
ssh jdh4@perseus-amd.princeton.edu
module load anaconda3/2019.10

OMP_NUM_THREADS=<num> python lu.py
or
MKL_DEBUG_CPU_TYPE=5 OMP_NUM_THREADS=<num> python lu.py
or
conda create --name nomkl-env numpy scipy nomkl
conda activate nomkl-env
OMP_NUM_THREADS=<num> python lu.py
```

| Machine       | OMP_NUM_THREADS | Execution time (s) |
|:-------------:|:---------------:|:------------------:|
| AMD Perseus   | 1               | 67.0 |
| AMD Perseus (a)   | 1               | 20.1 |
| Della Cascade | 1               | 12.0 |
| Della Haswell | 1               | 17.6 |
| AMD Perseus   | 2               | 34.9 |
| Della Cascade | 2               |  6.5 |
| Della Haswell | 2               | 10.0 |
| Della Haswell (a) | 2               | 10.3 |
| AMD Perseus   | 4               | 18.0 |
| AMD Perseus (a)   | 4               | 6.3 |
| AMD Perseus (b)   | 4               | 4.6 |
| Della Cascade | 4               |  3.9 |
| AMD Perseus   | 8               |  9.8 |
| AMD Perseus (a)   | 8               |  3.9 |
| AMD Perseus (b)   | 8               |  2.9 |
| Della Cascade | 8               |  2.8 |
| AMD Perseus   | 16              |  6.1 |
| Della Cascade | 16              |  2.2 |
| AMD Perseus   | 32              |  3.9 |
| Della Cascade | 32              |  1.9 |

The execution times are the best of 5 runs.

(a) OpenBLAS instead of MKL

(b) MKL_DEBUG_CPU_TYPE=5

Below is the Python script:

```python
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

| Machine     | Compiler  |MPI threads | OMP_NUM_THREADS | Execution time | Performance |
|:-----------:|:---------:|:-----------:|:---------------:|:--------------:|:-----------:|
| AMD Perseus | aocc      | 8           | 1               | 31.8               | 54.4 |
| AMD Perseus | intel     | 8           | 1               | 30.3               | 57.1 |
| tigerCpu    | intel     | 8           | 1               | 27.5               | 62.7 |
| AMD Perseus | aocc      | 16          | 1               | 19.6               | 88.2 |
| AMD Perseus | intel     | 16          | 1               | 19.5               | 88.7 |
| tigerCpu    | intel     | 16          | 1               | 15.7               | 109.9|
| Della Cascade| intel    | 16          | 1               | 15.8               | 109.3|
| tigerCpu    | intel     | 32          | 1               |  9.6               | 179.8|
| AMD Perseus | aocc      | 32          | 1               | 14.6               | 118.0|
| AMD Perseus | intel     | 32          | 1               | 13.4               | 128.6|
| AMD Perseus | aocc      | 64          | 1               | 9.7                | 177.8|
| AMD Perseus | intel     | 64          | 1               | 10.2               | 170.0|

[Build procedure](https://github.com/jdh4/running_gromacs/blob/master/02_installation/tigerCpu/tigerCpu.sh) for tigerCpu

Obtaining the benchmark:

```
wget ftp://ftp.gromacs.org/pub/benchmarks/rnase_bench_systems.tar.gz
tar -zxvf rnase_bench_systems.tar.gz
BCH=rnase_cubic
gmx grompp -f $BCH/pme_verlet.mdp -c $BCH/conf.gro -p $BCH/topol.top -o bench.tpr
gmx mdrun -ntomp 8 -s bench.tpr
gmx mdrun -ntmpi 16 -ntomp 1 -s bench.tpr
gmx mdrun -ntmpi 1 -ntomp 16 -s bench.tpr
```

### GROMACS with AOCC

GROMACS was built according to this procedure:

```bash
#!/bin/bash
version=2019.4
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version}.tar.gz
tar -zxvf gromacs-${version}.tar.gz
cd gromacs-${version}
mkdir build_stage1
cd build_stage1

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
# in future add -DGMX_OPENMP_MAX_THREADS=128

make
make check
make install
```

It is using the AVX2 instructions:

```
GROMACS version:    2019.4
Precision:          single
Memory model:       64 bit
MPI library:        thread_mpi
OpenMP support:     enabled (GMX_OPENMP_MAX_THREADS = 64)
GPU support:        disabled
SIMD instructions:  AVX2_256
FFT library:        fftw-3.3.8-sse2-avx-avx2-avx2_128-avx512
RDTSCP usage:       enabled
TNG support:        enabled
Hwloc support:      hwloc-1.11.8
Tracing support:    disabled
C compiler:         /opt/AMD/aocc-compiler-2.1.0/bin/clang Clang 9.0.0
C compiler flags:    -mavx2 -mfma     -Ofast -DNDEBUG
C++ compiler:       /opt/AMD/aocc-compiler-2.1.0/bin/clang++ Clang 9.0.0
C++ compiler flags:  -mavx2 -mfma    -std=c++11   -Ofast -DNDEBUG
```

### GROMACS with Intel

```
#!/bin/bash
version=2019.4
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-${version}.tar.gz
tar -zxvf gromacs-${version}.tar.gz
cd gromacs-${version}
mkdir build_stage1
cd build_stage1

module purge
module load intel/19.0/64/19.0.1.144

OPTFLAGS="-Ofast -DNDEBUG"

cmake3 .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS_RELEASE="$OPTFLAGS" \
-DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS_RELEASE="$OPTFLAGS" \
-DGMX_BUILD_MDRUN_ONLY=OFF -DGMX_MPI=OFF -DGMX_OPENMP=ON \
-DGMX_DOUBLE=OFF \
-DGMX_FFT_LIBRARY=mkl \
-DGMX_GPU=OFF \
-DCMAKE_INSTALL_PREFIX=$HOME/.local \
-DGMX_DEFAULT_SUFFIX=OFF -DGMX_BINARY_SUFFIX=_intel -DGMX_LIBS_SUFFIX=_intel \
-DGMX_COOL_QUOTES=OFF

make -j 10
make install
```

```
GROMACS version:    2019.4
Precision:          single
Memory model:       64 bit
MPI library:        thread_mpi
OpenMP support:     enabled (GMX_OPENMP_MAX_THREADS = 64)
GPU support:        disabled
SIMD instructions:  AVX2_256
FFT library:        Intel MKL
RDTSCP usage:       enabled
TNG support:        enabled
Hwloc support:      hwloc-1.11.8
Tracing support:    disabled
C compiler:         /opt/intel/compilers_and_libraries_2019.1.144/linux/bin/intel64/icc Intel 19.0.0.20181018
C compiler flags:    -march=core-avx2   -mkl=sequential  -std=gnu99  -Ofast -DNDEBUG -ip -funroll-all-loops -alias-const -ansi-alias -no-prec-div -fimf-domain-exclusion=14 -qoverride-limits  
C++ compiler:       /opt/intel/compilers_and_libraries_2019.1.144/linux/bin/intel64/icpc Intel 19.0.0.20181018
C++ compiler flags:  -march=core-avx2   -mkl=sequential  -std=c++11   -Ofast -DNDEBUG -ip -funroll-all-loops -alias-const -ansi-alias -no-prec-div -fimf-domain-exclusion=14 -qoverride-limits  
```
