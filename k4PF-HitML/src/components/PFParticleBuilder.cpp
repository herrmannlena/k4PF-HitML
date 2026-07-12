#include "PFParticleBuilder.h"

#include "Shower.h"

#include "edm4hep/Track.h"
#include "edm4hep/TrackState.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

static float massFromPredictedClass(int predictedClass) {
  switch (predictedClass) {
    case 0: return 0.000511f;   // electron
    case 1: return 0.139570f;   // charged hadron -> pion mass
    case 2: return 0.939565f;   // neutral hadron -> neutron mass
    case 3: return 0.f;         // photon
    case 4: return 0.105658f;   // muon
    default: return 0.f;
  }
}


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





static edm4hep::Vector3f momentumFromTrackState(const edm4hep::TrackState& ts, float bFieldTesla) {
  const float pt = 2.99792e-4f * std::abs(bFieldTesla / ts.omega);
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
  float bestScore = std::numeric_limits<float>::max();

  for (size_t i = 0; i < tracks.size(); ++i) {
    const float ndf = tracks[i].getNdf();
    const float score = (ndf > 0.f) ? tracks[i].getChi2() / ndf
                                    : std::numeric_limits<float>::max();

    if (score < bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return tracks[bestIdx];
}




static int chargeSignFromTrack(const edm4hep::TrackState& ts) {
  // Verify convention once with your sample.
  return (ts.omega > 0.f) ? +1 : -1;
}

// Energy-weighted position centroid of a set of calorimeter hits.
static edm4hep::Vector3f energyWeightedBarycenter(const std::vector<edm4hep::CalorimeterHit>& hits) {
  float sumE = 0.f, x = 0.f, y = 0.f, z = 0.f;
  for (const auto& hit : hits) {
    const auto pos = hit.getPosition();
    const float e = hit.getEnergy();
    sumE += e;
    x += pos.x * e;
    y += pos.y * e;
    z += pos.z * e;
  }
  if (sumE <= 0.f) {
    throw std::runtime_error("energyWeightedBarycenter: zero total calorimeter energy");
  }
  return {x / sumE, y / sumE, z / sumE};
}



ParticleRecoInfo buildChargedRecoInfo(
    const Shower& shower,
    int predictedClass,
    float pidScore,
    float bFieldTesla,
    bool reassignLowPMuons,
    float muonToChargedHadronPThreshold
) {
  ParticleRecoInfo out{};

  const auto& trk = pickBestTrack(shower);
  out.tracks = {trk};
  const auto& ts = trk.getTrackStates()[0];      // AtIP -- momentum, matches Python's pos_pxpypz_at_vertex
  const auto& tsCalo = trk.getTrackStates()[3];  // AtCalorimeter -- reference point

  const auto p3 = momentumFromTrackState(ts, bFieldTesla);
  const float p = momentumMagnitude(p3);

  //reassign low momentum muon to CH
  int effectiveClass = predictedClass;
  if (reassignLowPMuons && predictedClass == 4 && p < muonToChargedHadronPThreshold) {
    effectiveClass = 1;
  }
  const float mass = massFromPredictedClass(effectiveClass);


  // Reference point = energy-weighted shower barycenter minus the picked
  // track's calorimeter-entry position, matching Python's
  // PickPAtDCA.predict formula (tools_for_regression.py: "barycenters -
  // p_xyz"). Validated in Phase 3c to match numerically in raw mm.
  const auto barycenter = energyWeightedBarycenter(shower.getCalorimeterHits());

  out.momentum = p3;
  out.referencePoint = {
      barycenter.x - tsCalo.referencePoint.x,
      barycenter.y - tsCalo.referencePoint.y,
      barycenter.z - tsCalo.referencePoint.z
  };
  out.energy = p;
  out.pidScore = pidScore;
  out.physicsClass = effectiveClass;
  out.mass = mass;

  return out;
}


void fillRecoParticle(
    edm4hep::ReconstructedParticleCollection& pfParticles,
    edm4hep::ParticleIDCollection& pfParticleIDs,
    const Shower&,
    const ParticleRecoInfo& recoInfo
) {
  auto rp = pfParticles.create();
  auto pid = pfParticleIDs.create();

  rp.setMomentum(recoInfo.momentum);
  rp.setEnergy(recoInfo.energy);
  rp.setMass(recoInfo.mass);
  rp.setReferencePoint(recoInfo.referencePoint);

  for (const auto& trk : recoInfo.tracks) {
    rp.addToTracks(trk);
  }

  rp.setGoodnessOfPID(recoInfo.pidScore);


  pid.setLikelihood(recoInfo.pidScore);
  pid.setType(recoInfo.physicsClass);



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
  const auto& ecalHits = shower.ecalHits_;
  const auto& hcalHits = shower.hcalHits_;

  const float nEcal = static_cast<float>(ecalHits.size());
  const float nHcal = static_cast<float>(hcalHits.size());
  const float denom = nEcal + nHcal;

  // Match Python:
  // mask_ecal_only = (n_ecal_hits / (n_hcal_hits + n_ecal_hits)) > 0.05
  const bool useEcalOnly = (denom > 0.f) ? ((nEcal / denom) > 0.05f) : false;

  return useEcalOnly ? energyWeightedBarycenter(ecalHits)
                      : energyWeightedBarycenter(shower.getCalorimeterHits());
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

  const float mass = massFromPredictedClass(predictedClass);

  const float pMag = std::sqrt(std::max(0.f, predictedEnergy * predictedEnergy - mass * mass));

  out.momentum = {
      predictedDirection.x * pMag,
      predictedDirection.y * pMag,
      predictedDirection.z * pMag
  };
  out.referencePoint = predictedReferencePoint;
  out.direction = predictedDirection;

  out.energy = predictedEnergy;
  out.pidScore = pidScore;
  out.physicsClass = predictedClass;
  out.mass = mass;

  return out;
}







