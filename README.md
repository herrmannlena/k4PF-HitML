# k4PFHITML


describe + put paper link


## Dependencies

* ROOT

* PODIO

* Gaudi

* k4FWCore

## Installation

Run, from the `k4-project-template` directory:

``` bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
k4_local_repo
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -G Ninja -DPython_EXECUTABLE=$(which python3)
ninja install
```

Alternatively you can source the nightlies instead of the releases:

``` bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
```



## Execute Examples


``` bash
execute with k4run k4PF-HitML/options/PerformMLPF.py
```




## HitPF configurable parameters

describe