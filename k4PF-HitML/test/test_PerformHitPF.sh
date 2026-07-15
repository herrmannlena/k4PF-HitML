#!/bin/bash
set -e

# CMake's ExternalData resolves the DATA{} references in CMakeLists.txt to
# local, hash-verified file paths and passes them here as positional args.
# The input sample is not one of these -- it's regenerated fresh below with
# a fixed seed, same pattern as Tracking/test/testTrackFinder/test_trackFinder.sh.
MODEL_CLUSTERING=$1
MODEL_REGRESSION_NEUTRAL=$2
MODEL_REGRESSION_CHARGED=$3

SIM_FILE=k4PF-HitML/test/out_sim_edm4hep_test.root
OUTPUT_FILE=k4PF-HitML/test/output_HitPF_test.edm4hep.root
rm -f ${SIM_FILE} ${OUTPUT_FILE}

# k4geo's compact geometry files are part of the installed key4hep stack, so
# $K4GEO just works -- discovered dynamically here instead of hardcoding a
# version tag (e.g. CLD_o2_v05) that will go stale as k4geo evolves. If more
# than one CLD variant is installed this picks the first match; pin it
# explicitly (e.g. $K4GEO/FCCee/CLD/compact/CLD_o2_v05/CLD_o2_v05.xml) if that
# ever picks the wrong one for you.
XML_FILE=$(ls ${K4GEO}/FCCee/CLD/compact/*/*.xml | head -n1)

# Fetch just the CLD steering file (cld_steer.py) via curl -- CLDConfig is a
# much bigger repo than we need to clone for this one file, and the Tracking
# example's own test uses the same single-file curl approach for its (IDEA)
# steering file. -f makes curl fail loudly (rather than writing an HTML error
# page into the file) if the branch/path is wrong; if cld_steer.py turns out
# to import a companion module from elsewhere in CLDConfig, ddsim will fail
# with a clear ImportError below, telling us this needs a full clone after all.
STEERING_FILE=k4PF-HitML/test/cld_steer.py
if ! curl -fsSL -o "${STEERING_FILE}" https://raw.githubusercontent.com/key4hep/CLDConfig/main/CLDConfig/cld_steer.py; then
  curl -fsSL -o "${STEERING_FILE}" https://raw.githubusercontent.com/key4hep/CLDConfig/master/CLDConfig/cld_steer.py
fi

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
