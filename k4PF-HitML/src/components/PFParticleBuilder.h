#ifndef PFPARTICLEBUILDER_H
#define PFPARTICLEBUILDER_H

#include <algorithm>
#include <cmath>
#include <limits>


#include "edm4hep/Vector3f.h"
#include <vector>
#include "ONNXHelper.h" 
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/ParticleIDCollection.h"
#include "edm4hep/Track.h"

class Shower;

struct ParticleRecoInfo {
  edm4hep::Vector3f momentum{};
  edm4hep::Vector3f referencePoint{};
  edm4hep::Vector3f direction{};

  float energy{0.f};
  float mass{0.f};
  float pidScore{0.f};
  int physicsClass{0};
  float charge{0.f};
  int pdg{0};

  // Track(s) to attach to the output particle -- for charged candidates,
  // just the single track that drove momentum/mass (see pickBestTrack in
  // PFParticleBuilder.cpp), not every track that happened to land in the
  // same DPC cluster. Empty for neutral candidates.
  std::vector<edm4hep::Track> tracks{};

  std::vector<float> pidScores{};

};

struct PIDPrediction {
  int localIndex{0};     // index inside the NN output, e.g. 0,1,2
  int physicsClass{0};   // mapped class, e.g. 0,1,4 or 2,3
  float score{0.f};      // softmax probability of winning class
  std::vector<float> scores{};    // full softmax vector, one entry per class
                            
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
    float pidScore,
    float bFieldTesla = 2.0f,
    bool reassignLowPMuons = true,
    float muonToChargedHadronPThreshold = 1.0f
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



#endif
