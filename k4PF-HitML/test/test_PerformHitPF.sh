#!/bin/bash
set -e

# CMake's ExternalData resolves the DATA{} references in CMakeLists.txt to
# local, hash-verified file paths and passes them here as positional args.
# The input sample is not one of these -- it's regenerated fresh below with
# a fixed seed, same pattern as Tracking/test/testTrackFinder/test_trackFinder.sh.
MODEL_CLUSTERING=$1
MODEL_REGRESSION_NEUTRAL=$2
MODEL_REGRESSION_CHARGED=$3

NEV=5
SIM_FILE=k4PF-HitML/test/out_sim_edm4hep_test.root
REC_BASENAME=k4PF-HitML/test/out_reco_edm4hep_test
REC_FILE=${REC_BASENAME}_REC.edm4hep.root
OUTPUT_FILE=k4PF-HitML/test/output_HitPF_test.edm4hep.root
rm -f ${SIM_FILE} ${REC_FILE} ${OUTPUT_FILE}

# k4geo's compact geometry files are part of the installed key4hep stack, so
# $K4GEO just works. Pinned explicitly (not dynamically discovered) since CLD
# has multiple named geometry versions and cld_steer.py (fetched below) is
# only guaranteed to match a specific one -- "whichever version happens to be
# installed/sorts first" isn't safe here. Update CLDGEO if/when a newer
# CLDConfig steering file expects a different CLD version.
CLDGEO=CLD_o2_v07
XML_FILE=${K4GEO}/FCCee/CLD/compact/${CLDGEO}/${CLDGEO}.xml

# A single-file curl (like the Tracking example does for its one
# self-contained IDEA steering file) isn't enough here: CLDReconstruction.py
# imports a companion module (py_utils) from elsewhere in the CLDConfig repo,
# confirmed by the ModuleNotFoundError we got trying the single-file
# approach for it. So clone the whole repo instead, and reference both
# scripts from within it. Cloned once and reused on later runs (the
# `if [ ! -d ]` guard) -- not deleted afterwards: locally this makes repeat
# `ctest` runs during development skip the clone, and in real CI this whole
# build directory lives inside an ephemeral container that's discarded after
# the job anyway. Never commit this directory -- see .gitignore.
CLDCONFIG_DIR=k4PF-HitML/test/CLDConfig
if [ ! -d "${CLDCONFIG_DIR}" ]; then
  git clone --depth 1 https://github.com/key4hep/CLDConfig.git "${CLDCONFIG_DIR}"
fi
STEERING_FILE=${CLDCONFIG_DIR}/CLDConfig/cld_steer.py

ddsim --steeringFile ${STEERING_FILE} \
      --compactFile ${XML_FILE} \
      -G --gun.distribution uniform --gun.particle pi- \
      --random.seed 42 \
      --numberOfEvents ${NEV} \
      --outputFile ${SIM_FILE}

# CLDReconstruction.py requires its own directory as the working directory
# (it needs to find companion config there, e.g. Pandora settings XML) --
# confirmed by "Running Pandora is only possible when k4run is called from
# the directory: .../CLDConfig/CLDConfig". Absolute paths for input/output
# since they need to still resolve after the cd; a subshell so the cd
# doesn't affect the rest of this script (PerformMLPF.py's own paths below
# are relative to the original working directory).
SIM_FILE_ABS="$(pwd)/${SIM_FILE}"
REC_BASENAME_ABS="$(pwd)/${REC_BASENAME}"
(cd "${CLDCONFIG_DIR}/CLDConfig" && k4run CLDReconstruction.py -n ${NEV} \
  --inputFiles "${SIM_FILE_ABS}" \
  --outputBasename "${REC_BASENAME_ABS}")

k4run k4PF-HitML/options/PerformMLPF.py \
  --inputFiles "${REC_FILE}" \
  --onnx_model_clustering "${MODEL_CLUSTERING}" \
  --onnx_model_properties_neutral "${MODEL_REGRESSION_NEUTRAL}" \
  --onnx_model_properties_charged "${MODEL_REGRESSION_CHARGED}" \
  --outputFile ${OUTPUT_FILE} \
  --num_ev ${NEV}

python3 k4PF-HitML/test/check_output.py ${OUTPUT_FILE}
