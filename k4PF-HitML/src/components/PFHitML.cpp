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

 #include "k4FWCore/Transformer.h"
 #include "edm4hep/MCParticleCollection.h"
 #include <string>


//edm4hep imports
#include "edm4hep/TrackerHit.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"


//others
#include "ObservableExtractor.h"

 /**
  include description
  */



struct PFHitML final:
   k4FWCore::MultiTransformer<
   std::tuple<>(

    const edm4hep::MCParticleCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&

   )> {

    PFHitML(const std::string& name, ISvcLocator* svcLoc)
      : k4FWCore::MultiTransformer<
      std::tuple<>(
        const edm4hep::MCParticleCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&)>(
        name, svcLoc,
          {
            KeyValues("MCParticles", {"MCParticles"}),
            KeyValues("EcalBarrelHits", {"ECALBarrel"}),
            KeyValues("HcalBarrelHits", {"HCALBarrel"}),
            KeyValues("EcalBarrelHits", {"ECALEndcap"}),
            KeyValues("HcalBarrelHits", {"HCALEndcap"}),
            KeyValues("HcalOtherHits", {"HCALOther"}),
            KeyValues("MUON", {"MUON"})
          },
          {}  // no Outputs
        ) {}

  // main
  std::tuple<> operator()(
    const edm4hep::MCParticleCollection& mc_particles,
    const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalOther_hits,
    const edm4hep::CalorimeterHitCollection& Muon_hits
  ) const override {

    info() << "MCParticles: " << mc_particles.size() << endmsg;
    info() << "EcalBarrelHits: " << EcalBarrel_hits.size() << endmsg;
    info() << "HcalBarrelHits: " << HcalBarrel_hits.size() << endmsg;
    info() << "EcalEndcapHits: " << EcalEndcap_hits.size() << endmsg;
    info() << "HcalEndcapHits: " << HcalEndcap_hits.size() << endmsg;
    info() << "HcalOtherHits: " << HcalOther_hits.size() << endmsg;
    info() << "MuonHits: " << Muon_hits.size() << endmsg;

    ObservableExtractor extractor(
      mc_particles, 
      EcalBarrel_hits, 
      HcalBarrel_hits, 
      EcalEndcap_hits, 
      HcalEndcap_hits, 
      HcalOther_hits,
      Muon_hits
    );
    std::map<std::string, std::vector<float>> inputs = extractor.extract();


    // Iterate through the map and print key and length of each value
    for (const auto& pair : inputs) {
        std::cout << "Key: " << pair.first
                  << ", Length: " << pair.second.size() << std::endl;

      for (size_t i=0; i<10; i++){
                    std::cout << pair.second[i] << std::endl;
        }
    }

    



    return {}; // no outputs
  }
};


DECLARE_COMPONENT(PFHitML)