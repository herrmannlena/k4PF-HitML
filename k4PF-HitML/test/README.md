test_PerformMLPF (registered in k4PF-HitML/CMakeLists.txt) runs the full PerformMLPF.py pipeline on a small input sample with small test versions of the 3 ONNX models, then asserts (check_output.py) that the output HitPF collection is non-empty. This is not yet runnable as committed -- the 4 DATA{...} references in CMakeLists.txt point at files that don't exist in the repo yet:

k4PF-HitML/test/inputFiles/small_test_sample.edm4hep.root
k4PF-HitML/test/models/clustering_test.onnx
k4PF-HitML/test/models/regression_neutral_test.onnx
k4PF-HitML/test/models/regression_charged_test.onnx
CMake's ExternalData module never stores the actual binary content in git -- only a small hash sidecar file per fixture (e.g. small_test_sample.edm4hep.root.sha256, containing just the hex hash), and a URL template (set in the top-level CMakeLists.txt) telling CMake where to download the real content from at build/test time.

To add a real fixture:

Produce a small file (a few-event CLD sample from ddsim, or a truncated version of the current models re-exported to keep them small).
Compute its hash and save it as the sidecar file, e.g.:
sha256sum small_test_sample.edm4hep.root | awk '{print $1}' \
  > small_test_sample.edm4hep.root.sha256
Commit only the .sha256 file, not the real data file.
Upload the real file, named by its hash, to wherever ExternalData_URL_TEMPLATES points (currently https://key4hep.web.cern.ch:443/testFiles/k4PFHitML/%(hash) -- ask whoever administers that server to add it, the same way other key4hep packages' test fixtures got added).
Alternative for local-only testing without needing that upload step: set ExternalData_OBJECT_STORES to a local directory containing files named by hash, and skip the URL/upload entirely -- useful while developing the fixtures before they're hosted properly.