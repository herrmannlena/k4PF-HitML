# k4PFHitML CTest fixtures

`test_PerformHitPF` (registered in `k4PF-HitML/CMakeLists.txt`) runs the full
`PerformMLPF.py` pipeline on a tiny freshly-simulated sample with small test
versions of the 3 ONNX models, then asserts (`check_output.py`) that the
output `HitPF` collection is non-empty.


The 3 `DATA{...}` references in `CMakeLists.txt` are:

```
k4PF-HitML/test/models/clustering_test.onnx
k4PF-HitML/test/models/regression_neutral_test.onnx
k4PF-HitML/test/models/regression_charged_test.onnx
```


## Hosting the ONNX model fixtures

The real files live at `/eos/project/k/key4hep/www/key4hep/testFiles/k4PFHitML/`,
served at `https://key4hep.web.cern.ch:443/testFiles/k4PFHitML/%(hash)` (the
URL template in the top-level `CMakeLists.txt`). 

