#pragma once

#include "PFParticleBuilder.h"
#include "Shower.h"
#include "TruthMatcher.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

struct EvalRow {
  bool hasTruth{false};
  bool hasReco{false};
  bool isFake{false};
  bool isMissed{false};

  int showerIndex{-1};
  int truthIndex{-1};
  unsigned truthCollectionID{0};

  float trueEnergy{std::numeric_limits<float>::quiet_NaN()};
  int truePDG{0};
  float recoShowersE{std::numeric_limits<float>::quiet_NaN()};
  int isTrackInMC{0};
  int genStatus{0};

  float calibratedE{std::numeric_limits<float>::quiet_NaN()};
  float predShowersE{std::numeric_limits<float>::quiet_NaN()};
  int predClass{-1};
  int predPDG{0};
  float pidScore{std::numeric_limits<float>::quiet_NaN()};
  float iou{std::numeric_limits<float>::quiet_NaN()};

  float px{std::numeric_limits<float>::quiet_NaN()};
  float py{std::numeric_limits<float>::quiet_NaN()};
  float pz{std::numeric_limits<float>::quiet_NaN()};

  float refx{std::numeric_limits<float>::quiet_NaN()};
  float refy{std::numeric_limits<float>::quiet_NaN()};
  float refz{std::numeric_limits<float>::quiet_NaN()};

  float vx{std::numeric_limits<float>::quiet_NaN()};
  float vy{std::numeric_limits<float>::quiet_NaN()};
  float vz{std::numeric_limits<float>::quiet_NaN()};
};

class EvaluationSummaryBuilder {
public:
  EvaluationSummaryBuilder(
      const std::vector<Shower>& showers,
      const std::vector<ShowerTruthMatch>& matches,
      const std::vector<TruthRecoSummary>& truthSummaries
  );

  void addRecoResult(std::size_t showerIdx, const ParticleRecoInfo& recoInfo);

  std::vector<EvalRow> finalize() const;

private:
  const std::vector<Shower>& m_showers;
  const std::vector<ShowerTruthMatch>& m_matches;
  const std::vector<TruthRecoSummary>& m_truthSummaries;

  std::vector<std::optional<ParticleRecoInfo>> m_recoInfoPerShower;
  std::vector<float> m_predClusterEnergyPerShower;
};
