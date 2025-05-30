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
 #ifndef OBSERVABLEEXTRACTOR_H
 #define OBSERVABLEEXTRACTOR_H

 //edm4hep imports
#include "edm4hep/TrackerHit.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"
#include "edm4hep/MCParticleCollection.h"
 
 class ObservableExtractor {
 public:
    ObservableExtractor(const edm4hep::MCParticleCollection& mc_particles,
    const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalEndcap_hits);

    std::map<std::string, std::vector<float>> extract() const;

 private:
    const edm4hep::MCParticleCollection& mc_;
    const edm4hep::CalorimeterHitCollection& ecalbarrel_;
    const edm4hep::CalorimeterHitCollection& hcalbarrel_;
    const edm4hep::CalorimeterHitCollection& ecalendcap_;
    const edm4hep::CalorimeterHitCollection& hcalendcap_;

    //was private, was public?

  };
 
  #endif // OBSERVABLEEXTRACTOR_H