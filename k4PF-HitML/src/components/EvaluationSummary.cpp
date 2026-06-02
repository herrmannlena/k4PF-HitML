#include "EvaluationSummary.h"

#include <stdexcept>
#include <unordered_map>

EvaluationSummaryBuilder::EvaluationSummaryBuilder(
    const std::vector<Shower>& showers,
    const std::vector<ShowerTruthMatch>& matches,
    const std::vector<TruthRecoSummary>& truthSummaries
)
    : m_showers(showers),
      m_matches(matches),
      m_truthSummaries(truthSummaries),
      m_recoInfoPerShower(showers.size()),
      m_predClusterEnergyPerShower(showers.size(), 0.f) {}

void EvaluationSummaryBuilder::addRecoResult(
    std::size_t showerIdx,
    const ParticleRecoInfo& recoInfo
) {
  if (showerIdx >= m_showers.size()) {
    throw std::out_of_range("showerIdx out of range in EvaluationSummaryBuilder");
  }

  m_recoInfoPerShower[showerIdx] = recoInfo;
  m_predClusterEnergyPerShower[showerIdx] =
      m_showers[showerIdx].getCaloEnergy(m_showers[showerIdx].caloHits_).first;
}

std::vector<EvalRow> EvaluationSummaryBuilder::finalize() const {
  std::vector<EvalRow> evalRows;
  std::unordered_map<ObjectIDKey, std::size_t, ObjectIDKeyHash> truthRowIndex;

  for (const auto& ts : m_truthSummaries) {
    EvalRow row;
    row.hasTruth = true;
    row.isMissed = true;

    row.truthIndex = ts.key.index;
    row.truthCollectionID = ts.key.collectionID;
    row.trueEnergy = ts.mc.getEnergy();
    //row.truePDG = ts.mc.getPDG();
    row.recoShowersE = ts.recoCaloEnergy;
    row.isTrackInMC = ts.nTracks;
    row.genStatus = ts.mc.getGeneratorStatus();

    const auto v = ts.mc.getVertex();
    row.vx = static_cast<float>(v.x);
    row.vy = static_cast<float>(v.y);
    row.vz = static_cast<float>(v.z);

    truthRowIndex[ts.key] = evalRows.size();
    evalRows.push_back(row);
  }

  for (std::size_t idx = 0; idx < m_showers.size(); ++idx) {
    if (!m_recoInfoPerShower[idx].has_value()) {
      continue;
    }

    const auto& reco = *m_recoInfoPerShower[idx];
    const auto& match = m_matches[idx];

    if (match.matched) {
      auto it = truthRowIndex.find(match.key);
      if (it == truthRowIndex.end()) {
        continue;
      }

      auto& row = evalRows[it->second];
      row.hasReco = true;
      row.isMissed = false;
      row.showerIndex = static_cast<int>(idx);

      row.calibratedE = reco.energy;
      row.predShowersE = m_predClusterEnergyPerShower[idx];
      row.predClass = reco.physicsClass;
      //row.predPDG = reco.pdg;
      row.pidScore = reco.pidScore;
      row.iou = match.iou;

      row.px = reco.momentum.x;
      row.py = reco.momentum.y;
      row.pz = reco.momentum.z;

      row.refx = reco.referencePoint.x;
      row.refy = reco.referencePoint.y;
      row.refz = reco.referencePoint.z;
    } else {
      EvalRow row;
      row.hasReco = true;
      row.isFake = true;
      row.showerIndex = static_cast<int>(idx);

      row.calibratedE = reco.energy;
      row.predShowersE = m_predClusterEnergyPerShower[idx];
      row.predClass = reco.physicsClass;
      //row.predPDG = reco.pdg;
      row.pidScore = reco.pidScore;

      row.px = reco.momentum.x;
      row.py = reco.momentum.y;
      row.pz = reco.momentum.z;

      row.refx = reco.referencePoint.x;
      row.refy = reco.referencePoint.y;
      row.refz = reco.referencePoint.z;

      evalRows.push_back(row);
    }
  }

  return evalRows;
}
