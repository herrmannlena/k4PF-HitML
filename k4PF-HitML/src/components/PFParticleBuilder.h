#include <algorithm>
#include <cmath>
#include <limits>


#include "edm4hep/Vector3f.h"
#include <vector>
#include "ONNXHelper.h" 
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/ParticleIDCollection.h"

class Shower;

struct ParticleRecoInfo {
  edm4hep::Vector3f momentum{};
  edm4hep::Vector3f referencePoint{};

  float energy{0.f};
  float mass{0.f};
  float charge{0.f};

  int pdg{0};
  float pidScore{0.f};
};

struct PIDPrediction {
  int localIndex{0};     // index inside the NN output, e.g. 0,1,2
  int physicsClass{0};   // mapped class, e.g. 0,1,4 or 2,3
  float score{0.f};      // softmax probability of winning class
};

const std::vector<float>& findPIDOutput(
    const ONNXHelper& model,
    const std::vector<std::vector<float>>& outputs
);


PIDPrediction decodePIDLogits(
    const std::vector<float>& logits,
    const std::vector<int>& classMap
);


ParticleRecoInfo buildChargedRecoInfo(
    const Shower& shower,
    int predictedClass,
    float pidScore
);

ParticleRecoInfo buildNeutralRecoInfo(
    const Shower& shower,
    int predictedClass,
    float pidScore,
    float predictedEnergy,
    const edm4hep::Vector3f& predictedDirection,
    const edm4hep::Vector3f& predictedReferencePoint
);

edm4hep::Vector3f computeNeutralReferencePoint(const Shower& shower);


edm4hep::Vector3f computeNeutralDirection(const edm4hep::Vector3f& referencePoint);


void fillRecoParticle(
    edm4hep::ReconstructedParticleCollection& particles,
    edm4hep::ParticleIDCollection& pids,
    const Shower& shower,
    const ParticleRecoInfo& info
);


