#!/bin/bash
set -e

# CMake's ExternalData resolves the DATA{} references in CMakeLists.txt to
# local, hash-verified file paths and passes them here as positional args.
INPUT_FILE=$1
MODEL_CLUSTERING=$2
MODEL_REGRESSION_NEUTRAL=$3
MODEL_REGRESSION_CHARGED=$4

OUTPUT_FILE=k4PF-HitML/test/output_HitPF_test.root
rm -f ${OUTPUT_FILE}

k4run k4PF-HitML/options/PerformMLPF.py \
  --inputFiles "${INPUT_FILE}" \
  --onnx_model_clustering "${MODEL_CLUSTERING}" \
  --onnx_model_properties_neutral "${MODEL_REGRESSION_NEUTRAL}" \
  --onnx_model_properties_charged "${MODEL_REGRESSION_CHARGED}" \
  --outputFile ${OUTPUT_FILE} \
  --num_ev 5

python3 k4PF-HitML/test/check_output.py ${OUTPUT_FILE}
