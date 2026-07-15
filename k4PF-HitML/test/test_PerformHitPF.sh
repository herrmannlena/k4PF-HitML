#!/bin/bash
set -e

# CMake's ExternalData resolves the DATA{} references in CMakeLists.txt to
# local, hash-verified file paths and passes them here as positional args.

MODEL_CLUSTERING=$1
MODEL_REGRESSION_NEUTRAL=$2
MODEL_REGRESSION_CHARGED=$3

SIM_FILE=k4PF-HitML/test/out_sim_edm4hep_test.root
OUTPUT_FILE=k4PF-HitML/test/output_HitPF_test.root
rm -f ${SIM_FILE} ${OUTPUT_FILE}

XML_FILE=$(ls ${K4GEO}/FCCee/CLD/compact/*/*.xml | head -n1)


CLDCONFIG_DIR=k4PF-HitML/test/CLDConfig
if [ ! -d "${CLDCONFIG_DIR}" ]; then
  git clone --depth 1 https://github.com/key4hep/CLDConfig.git "${CLDCONFIG_DIR}"
fi
STEERING_FILE=${CLDCONFIG_DIR}/CLDConfig/cld_steer.py

ddsim --steeringFile ${STEERING_FILE} \
      --compactFile ${XML_FILE} \
      -G --gun.distribution uniform --gun.particle pi- \
      --random.seed 42 \
      --numberOfEvents 5 \
      --outputFile ${SIM_FILE}

k4run k4PF-HitML/options/PerformMLPF.py \
  --inputFiles "${SIM_FILE}" \
  --onnx_model_clustering "${MODEL_CLUSTERING}" \
  --onnx_model_properties_neutral "${MODEL_REGRESSION_NEUTRAL}" \
  --onnx_model_properties_charged "${MODEL_REGRESSION_CHARGED}" \
  --outputFile ${OUTPUT_FILE} \
  --num_ev 5

python3 k4PF-HitML/test/check_output.py ${OUTPUT_FILE}
