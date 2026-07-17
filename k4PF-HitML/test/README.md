<!--
Copyright (c) 2020-2024 Key4hep-Project.

This file is part of Key4hep.
See https://key4hep.github.io/key4hep-doc/ for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# k4PFHitML CTest fixtures

`test_PerformHitPF` (registered in `k4PF-HitML/CMakeLists.txt`) runs the full
`PerformMLPF.py` pipeline on a tiny simulated sample with small test
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

