#include "PFParticleBuilder.h"

#include "Shower.h"

#include "edm4hep/Track.h"
#include "edm4hep/TrackState.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>


const std::vector<float>& findPIDOutput(
    const ONNXHelper& model,
    const std::vector<std::vector<float>>& outputs
) {
  const auto& names = model.outputNames();

  for (size_t i = 0; i < names.size() && i < outputs.size(); ++i) {
    if (names[i].find("pid") != std::string::npos) {
      return outputs[i];
    }
  }

  throw std::runtime_error("Could not find PID output tensor");
}

PIDPrediction decodePIDLogits(
    const std::vector<float>& logits,
    const std::vector<int>& classMap
) {
  if (logits.empty()) {
    throw std::runtime_error("PID logits are empty");
  }

  if (logits.size() != classMap.size()) {
    throw std::runtime_error("PID logits size does not match class map size");
  }

  const auto bestIt = std::max_element(logits.begin(), logits.end());
  const int localIndex = static_cast<int>(std::distance(logits.begin(), bestIt));

  const float maxLogit = *bestIt;
  float sumExp = 0.f;
  for (float x : logits) {
    sumExp += std::exp(x - maxLogit);
  }

  PIDPrediction pred;
  pred.localIndex = localIndex;
  pred.physicsClass = classMap[localIndex];
  pred.score = (sumExp > 0.f) ? 1.f / sumExp : 0.f;
  return pred;
}


namespace {


static edm4hep::Vector3f momentumFromTrackState(const edm4hep::TrackState& ts) {
  const float pt = 2.99792e-4f * std::abs(2.f / ts.omega); // for B=2T
  const float px = std::cos(ts.phi) * pt;
  const float py = std::sin(ts.phi) * pt;
  const float pz = ts.tanLambda * pt;
  return {px, py, pz};
}

static float momentumMagnitude(const edm4hep::Vector3f& p) {
  return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

static const edm4hep::Track& pickBestTrack(const Shower& shower) {
  const auto& tracks = shower.getTracks();
  if (tracks.empty()) {
    throw std::runtime_error("Charged shower has no track");
  }

  size_t bestIdx = 0;
  float bestChi2 = std::numeric_limits<float>::max();
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (tracks[i].getChi2() < bestChi2) {
      bestChi2 = tracks[i].getChi2();
      bestIdx = i;
    }
  }
  return tracks[bestIdx];
}

static int chargeSignFromTrack(const edm4hep::TrackState& ts) {
  // Verify convention once with your sample.
  return (ts.omega > 0.f) ? +1 : -1;
}

static float massFromPDG(int pdg) {
  switch (std::abs(pdg)) {
    case 11:  return 0.000511f;
    case 13:  return 0.105658f;
    case 211: return 0.139570f;
    default:  return 0.139570f; // charged-hadron bucket -> pion mass
  }
}

static int chargedClassToPDG(int predictedClass, int chargeSign) {
  // Python charged classes are [0, 1, 4]
  // 0 -> electron, 1 -> charged hadron, 4 -> muon
  if (predictedClass == 0) {
    return chargeSign > 0 ? -11 : 11;
  }
  if (predictedClass == 4) {
    return chargeSign > 0 ? -13 : 13;
  }
  return chargeSign > 0 ? 211 : -211;
}

} // namespace



ParticleRecoInfo buildChargedRecoInfo(
    const Shower& shower,
    int predictedClass,
    float pidScore
) {
  ParticleRecoInfo out{};

  const auto& trk = pickBestTrack(shower);
  const auto& ts = trk.getTrackStates()[1]; // correct index??

  const auto p3 = momentumFromTrackState(ts);
  const float p = momentumMagnitude(p3);
  const int q = chargeSignFromTrack(ts);
  const int pdg = chargedClassToPDG(predictedClass, q);
  const float mass = massFromPDG(pdg);
  const float energy = std::sqrt(p * p + mass * mass);

  //debugging
  //std::cout << "  predictedClass = " << predictedClass << std::endl;
  //std::cout << "  pidScore       = " << pidScore << std::endl;
  //std::cout << "  nTracks        = " << shower.getTracks().size() << std::endl;
  //std::cout << "  |p|    = " << p << std::endl;
  //std::cout << "  charge = " << q << std::endl;
  //std::cout << "  pdg    = " << pdg << std::endl;
  //std::cout << "  mass   = " << mass << std::endl;
  //std::cout << "  energy = " << energy << std::endl;


  out.momentum = p3;
  out.referencePoint = ts.referencePoint;
  out.energy = energy;
  out.mass = mass;
  out.charge = static_cast<float>(q);
  out.pdg = pdg;
  out.pidScore = pidScore;
  return out;
}

void fillRecoParticle(
    edm4hep::ReconstructedParticleCollection& pfParticles,
    edm4hep::ParticleIDCollection& pfParticleIDs,
    const Shower& shower,
    const ParticleRecoInfo& recoInfo
) {
  auto rp = pfParticles.create();
  auto pid = pfParticleIDs.create();

  rp.setMomentum(recoInfo.momentum);
  rp.setEnergy(recoInfo.energy);
  rp.setMass(recoInfo.mass);
  rp.setCharge(recoInfo.charge);
  rp.setReferencePoint(recoInfo.referencePoint);

  for (const auto& trk : shower.getTracks()) {
    rp.addToTracks(trk);
  }

  pid.setPDG(recoInfo.pdg);
  pid.setLikelihood(recoInfo.pidScore);


}


edm4hep::Vector3f computeNeutralDirection(const edm4hep::Vector3f& referencePoint) {
  
  const float norm = std::sqrt(
      referencePoint.x * referencePoint.x +
      referencePoint.y * referencePoint.y +
      referencePoint.z * referencePoint.z
  );

  if (norm <= 0.f) {
    throw std::runtime_error("Neutral reference point has zero norm");
  }

  return {
      referencePoint.x / norm,
      referencePoint.y / norm,
      referencePoint.z / norm
  };
}

edm4hep::Vector3f computeNeutralReferencePoint(const Shower& shower) {
  float sumE = 0.f;
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;

  for (const auto& hit : shower.getCalorimeterHits()) {
    const auto pos = hit.getPosition();
    const float e = hit.getEnergy();
    sumE += e;
    x += pos.x * e;
    y += pos.y * e;
    z += pos.z * e;
  }

  if (sumE <= 0.f) {
    throw std::runtime_error("Neutral shower has zero calorimeter energy");
  }

  return {x / sumE, y / sumE, z / sumE};
}


ParticleRecoInfo buildNeutralRecoInfo(
    const Shower&,
    int predictedClass,
    float pidScore,
    float predictedEnergy,
    const edm4hep::Vector3f& predictedDirection,
    const edm4hep::Vector3f& predictedReferencePoint
) {
  ParticleRecoInfo out{};

  int pdg = 0;
  float mass = 0.f;

  // neutral classes: [2, 3]
  // 2 -> neutral hadron, 3 -> photon
  if (predictedClass == 3) {
    pdg = 22;
    mass = 0.f;
  } else {
    pdg = 130;
    mass = 0.497611f;
  }

  const float p2 = std::max(0.f, predictedEnergy * predictedEnergy - mass * mass);
  const float p = std::sqrt(p2);

    //debugging
  std::cout << "  predictedClass = " << predictedClass << std::endl;
  std::cout << "  pidScore       = " << pidScore << std::endl;
  std::cout << "  |p|    = " << p << std::endl;
  std::cout << "  pdg    = " << pdg << std::endl;
  std::cout << "  mass   = " << mass << std::endl;
  std::cout << "  energy = " << predictedEnergy << std::endl;

  out.momentum = {
      predictedDirection.x * p,
      predictedDirection.y * p,
      predictedDirection.z * p
  };
  out.referencePoint = predictedReferencePoint;
  out.energy = predictedEnergy;
  out.mass = mass;
  out.charge = 0.f;
  out.pdg = pdg;
  out.pidScore = pidScore;
  return out;
}


