/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
