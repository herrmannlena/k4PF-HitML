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

 #include "ObservableExtractor.h"

//edm4hep imports
#include "edm4hep/TrackerHit.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"

ObservableExtractor::ObservableExtractor(
    const edm4hep::MCParticleCollection& mc_particles,
    const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& HcalBarrel_hits)
    : mc_(mc_particles), ecalbarrel_(EcalBarrel_hits), hcalbarrel_(HcalBarrel_hits) {}
  

std::map<std::string, std::vector<float>> ObservableExtractor::extract() const {
    std::map<std::string, std::vector<float>> features;

    //stelle dir deine collections zusammen: zB aus calorimeterhits und appende hier
  
    // Example features:
    features["hit_px"].push_back(static_cast<float>(100));
    features["hit_py"].push_back(static_cast<float>(100));
    features["hit_pz"].push_back(static_cast<float>(100));
  
  
    return features;
  }