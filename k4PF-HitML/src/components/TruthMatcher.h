#pragma once

#include "Shower.h"

#include "edm4hep/CaloHitMCParticleLinkCollection.h"
#include "edm4hep/MCParticle.h"
#include "edm4hep/RecoMCParticleLinkCollection.h"
#include "edm4hep/ReconstructedParticle.h"
#include "edm4hep/TrackMCParticleLinkCollection.h"
#include "podio/ObjectID.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct TruthMatchConfig {
  float iouThreshold{0.25f};
  float barrelRadius{2150.f};
  int nBarrelSides{12};
  float endCapZ{2307.f};
};

struct ObjectIDKey {
  int index{-1};
  unsigned collectionID{0};

  bool operator==(const ObjectIDKey& other) const {
    return index == other.index && collectionID == other.collectionID;
  }

  bool valid() const {
    return index >= 0;
  }
};

struct ObjectIDKeyHash {
  std::size_t operator()(const ObjectIDKey& key) const {
    const auto a = static_cast<std::uint64_t>(key.collectionID);
    const auto b = static_cast<std::uint32_t>(key.index);
    return std::hash<std::uint64_t>{}((a << 32) ^ b);
  }
};

inline ObjectIDKey makeKey(const podio::ObjectID& id) {
  return ObjectIDKey{id.index, id.collectionID};
}

struct TruthLabel {
  bool valid{false};
  edm4hep::MCParticle mc{};
  ObjectIDKey key{};
};

struct ShowerTruthMatch {
  bool matched{false};
  edm4hep::MCParticle mc{};
  ObjectIDKey key{};
  float iou{0.f};
  float sharedEnergy{0.f};
};

namespace truth_detail {

constexpr int BIT_BACKSCATTER = 29;
constexpr int BIT_DECAYED_IN_TRACKER = 27;

inline bool checkBit(int value, int bit) {
  return ((value >> bit) & 1) == 1;
}

inline bool backscatteredAndTracker(int simStatus) {
  const bool decayedInTracker = checkBit(simStatus, BIT_DECAYED_IN_TRACKER);
  const bool backscatter = checkBit(simStatus, BIT_BACKSCATTER);
  return backscatter && !decayedInTracker;
}

inline bool isProducedInCalo(
    const edm4hep::Vector3d& vtx,
    float barrelRadius,
    int nBarrelSides,
    float endCapZ
) {
  bool producedInCalo = false;
  for (int i = 0; i < nBarrelSides; ++i) {
    const double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(nBarrelSides);
    const double nx = std::cos(angle);
    const double ny = std::sin(angle);
    if (vtx.x * nx + vtx.y * ny > barrelRadius) {
      producedInCalo = true;
      break;
    }
  }
  if (std::abs(vtx.z) > endCapZ) {
    producedInCalo = true;
  }
  return producedInCalo;
}

inline edm4hep::MCParticle remapToCaloMother(
    edm4hep::MCParticle mc,
    const TruthMatchConfig& cfg
) {
  while (mc.isAvailable()) {
    const bool producedInCalo = isProducedInCalo(
        mc.getVertex(), cfg.barrelRadius, cfg.nBarrelSides, cfg.endCapZ
    );
    const bool keepAsIs = backscatteredAndTracker(mc.getSimulatorStatus());

    if (!producedInCalo || keepAsIs) {
      break;
    }

    const auto parents = mc.getParents();
    if (parents.empty()) {
      break;
    }
    mc = parents[0];
  }
  return mc;
}

template <typename LinkCollectionT>
std::unordered_map<ObjectIDKey, TruthLabel, ObjectIDKeyHash>
buildBestTruthMap(
    const LinkCollectionT& links,
    bool remapCaloMother,
    const TruthMatchConfig& cfg
) {
  struct BestLinkInfo {
    float weight{-std::numeric_limits<float>::infinity()};
    edm4hep::MCParticle mc{};
  };

  std::unordered_map<ObjectIDKey, BestLinkInfo, ObjectIDKeyHash> bestPerObject;

  for (const auto& link : links) {
    const auto fromKey = makeKey(link.getFrom().getObjectID());
    const float w = link.getWeight();

    auto it = bestPerObject.find(fromKey);
    if (it == bestPerObject.end() || w > it->second.weight) {
      bestPerObject[fromKey] = BestLinkInfo{w, link.getTo()};
    }
  }

  std::unordered_map<ObjectIDKey, TruthLabel, ObjectIDKeyHash> result;
  result.reserve(bestPerObject.size());

  for (const auto& [objKey, best] : bestPerObject) {
    edm4hep::MCParticle mc = best.mc;
    if (remapCaloMother) {
      mc = remapToCaloMother(mc, cfg);
    }
    result[objKey] = TruthLabel{mc.isAvailable(), mc, makeKey(mc.getObjectID())};
  }

  return result;
}

template <typename T>
std::vector<T> uniqueVector(std::vector<T> v) {
  std::sort(v.begin(), v.end(), [](const T& a, const T& b) {
    if (a.index != b.index) {
      return a.index < b.index;
    }
    return a.collectionID < b.collectionID;
  });
  v.erase(std::unique(v.begin(), v.end(), [](const T& a, const T& b) {
    return a.index == b.index && a.collectionID == b.collectionID;
  }), v.end());
  return v;
}

// Hungarian maximization by converting to minimization cost.
inline std::vector<std::pair<int, int>> hungarianMaximize(
    const std::vector<std::vector<float>>& score
) {
  const int nRows = static_cast<int>(score.size());
  const int nCols = (nRows > 0) ? static_cast<int>(score[0].size()) : 0;
  const int n = std::max(nRows, nCols);

  if (nRows == 0 || nCols == 0) {
    return {};
  }

  float maxVal = 0.f;
  for (const auto& row : score) {
    for (float x : row) {
      maxVal = std::max(maxVal, x);
    }
  }

  std::vector<std::vector<float>> cost(n + 1, std::vector<float>(n + 1, maxVal));
  for (int i = 1; i <= nRows; ++i) {
    for (int j = 1; j <= nCols; ++j) {
      cost[i][j] = maxVal - score[i - 1][j - 1];
    }
  }

  std::vector<float> u(n + 1), v(n + 1);
  std::vector<int> p(n + 1), way(n + 1);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<float> minv(n + 1, std::numeric_limits<float>::infinity());
    std::vector<char> used(n + 1, false);

    do {
      used[j0] = true;
      const int i0 = p[j0];
      float delta = std::numeric_limits<float>::infinity();
      int j1 = 0;

      for (int j = 1; j <= n; ++j) {
        if (used[j]) {
          continue;
        }
        const float cur = cost[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }

      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<std::pair<int, int>> assignment;
  for (int j = 1; j <= n; ++j) {
    if (p[j] >= 1 && p[j] <= nRows && j >= 1 && j <= nCols) {
      assignment.emplace_back(p[j] - 1, j - 1);
    }
  }
  return assignment;
}

} // namespace truth_detail

template <typename CaloLinkCollectionT, typename TrackLinkCollectionT>
std::vector<ShowerTruthMatch> matchShowersByIoU(
    const std::vector<Shower>& showers,
    const CaloLinkCollectionT& caloTruthLinks,
    const TrackLinkCollectionT& trackTruthLinks,
    const TruthMatchConfig& cfg = {}
) {
  const auto hitTruth = truth_detail::buildBestTruthMap(caloTruthLinks, true, cfg);
  const auto trackTruth = truth_detail::buildBestTruthMap(trackTruthLinks, false, cfg);

  const ObjectIDKey backgroundKey{-1, 0};

  std::vector<std::unordered_map<ObjectIDKey, int, ObjectIDKeyHash>> intersections(showers.size());
  std::vector<std::unordered_map<ObjectIDKey, float, ObjectIDKeyHash>> intersectionsE(showers.size());
  std::unordered_map<ObjectIDKey, int, ObjectIDKeyHash> truthSizes;
  std::unordered_map<ObjectIDKey, edm4hep::MCParticle, ObjectIDKeyHash> truthParticles;
  std::vector<int> predSizes(showers.size(), 0);

  for (size_t i = 0; i < showers.size(); ++i) {
    const auto& shower = showers[i];

    for (const auto& hit : shower.getCalorimeterHits()) {
      ++predSizes[i];
      const auto key = makeKey(hit.getObjectID());
      auto it = hitTruth.find(key);
      const ObjectIDKey truthKey = (it != hitTruth.end() && it->second.valid) ? it->second.key : backgroundKey;

      intersections[i][truthKey] += 1;
      intersectionsE[i][truthKey] += hit.getEnergy();
      truthSizes[truthKey] += 1;

      if (truthKey.valid() && it != hitTruth.end()) {
        truthParticles[truthKey] = it->second.mc;
      }
    }

    for (const auto& trk : shower.getTracks()) {
      ++predSizes[i];
      const auto key = makeKey(trk.getObjectID());
      auto it = trackTruth.find(key);
      const ObjectIDKey truthKey = (it != trackTruth.end() && it->second.valid) ? it->second.key : backgroundKey;

      intersections[i][truthKey] += 1;
      truthSizes[truthKey] += 1;

      if (truthKey.valid() && it != trackTruth.end()) {
        truthParticles[truthKey] = it->second.mc;
      }
    }
  }

  std::vector<ObjectIDKey> truthKeys;
  truthKeys.reserve(truthSizes.size());
  for (const auto& [k, _] : truthSizes) {
    if (k.valid()) {
      truthKeys.push_back(k);
    }
  }
  truthKeys = truth_detail::uniqueVector(std::move(truthKeys));

  if (showers.empty() || truthKeys.empty()) {
    return std::vector<ShowerTruthMatch>(showers.size());
  }

  std::vector<std::vector<float>> iou(truthKeys.size(), std::vector<float>(showers.size(), 0.f));

  for (size_t t = 0; t < truthKeys.size(); ++t) {
    const auto& truthKey = truthKeys[t];
    const int truthSize = truthSizes[truthKey];

    for (size_t s = 0; s < showers.size(); ++s) {
      const auto it = intersections[s].find(truthKey);
      const int inter = (it != intersections[s].end()) ? it->second : 0;
      const int uni = predSizes[s] + truthSize - inter;
      if (uni > 0) {
        iou[t][s] = static_cast<float>(inter) / static_cast<float>(uni);
        if (iou[t][s] < cfg.iouThreshold) {
          iou[t][s] = 0.f;
        }
      }
    }
  }

  const auto assignment = truth_detail::hungarianMaximize(iou);

  std::vector<ShowerTruthMatch> result(showers.size());

  for (const auto& [truthRow, showerCol] : assignment) {
    const float score = iou[truthRow][showerCol];
    if (score <= 0.f) {
      continue;
    }

    const auto& truthKey = truthKeys[truthRow];
    result[showerCol].matched = true;
    result[showerCol].key = truthKey;
    result[showerCol].mc = truthParticles.at(truthKey);
    result[showerCol].iou = score;

    const auto eIt = intersectionsE[showerCol].find(truthKey);
    if (eIt != intersectionsE[showerCol].end()) {
      result[showerCol].sharedEnergy = eIt->second;
    }
  }

  return result;
}

inline void fillRecoTruthLink(
    edm4hep::RecoMCParticleLinkCollection& outLinks,
    const edm4hep::ReconstructedParticle& reco,
    const ShowerTruthMatch& match
) {
  if (!match.matched) {
    return;
  }

  auto link = outLinks.create();
  link.setFrom(reco);
  link.setTo(match.mc);
  link.setWeight(match.iou);
}
