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
#include "ONNXHelper.h"  
#include "Helpers.h"  
#include "ROOT/RVec.hxx"
#include <nlohmann/json.hpp> 

namespace rv = ROOT::VecOps;

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
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::TrackCollection&

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
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::TrackCollection&)>(
        name, svcLoc,
          {
            KeyValues("MCParticles", {"MCParticles"}),
            KeyValues("EcalBarrelHits", {"ECALBarrel"}),
            KeyValues("HcalBarrelHits", {"HCALBarrel"}),
            KeyValues("EcalEndcaplHits", {"ECALEndcap"}),
            KeyValues("HcalEndcapHits", {"HCALEndcap"}),
            KeyValues("HcalOtherHits", {"HCALOther"}),
            KeyValues("MUON", {"MUON"}),
            KeyValues("Tracks", {"SiTracks_Refitted"})
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
    const edm4hep::CalorimeterHitCollection& Muon_hits,
    const edm4hep::TrackCollection& tracks
  ) const override {

    info() << "MCParticles: " << mc_particles.size() << endmsg;
    info() << "EcalBarrelHits: " << EcalBarrel_hits.size() << endmsg;
    info() << "HcalBarrelHits: " << HcalBarrel_hits.size() << endmsg;
    info() << "EcalEndcapHits: " << EcalEndcap_hits.size() << endmsg;
    info() << "HcalEndcapHits: " << HcalEndcap_hits.size() << endmsg;
    info() << "HcalOtherHits: " << HcalOther_hits.size() << endmsg;
    info() << "MuonHits: " << Muon_hits.size() << endmsg;
    info() << "tracks: " << tracks.size() << endmsg;

    ObservableExtractor extractor(
      mc_particles, 
      EcalBarrel_hits, 
      HcalBarrel_hits, 
      EcalEndcap_hits, 
      HcalEndcap_hits, 
      HcalOther_hits,
      Muon_hits,
      tracks
   //   calo_truthlinks
    );
    std::map<std::string, std::vector<float>> inputs = extractor.extract();

    //still need to normalize etc


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

  StatusCode initialize() override {
    info() << "Initializing PFHitML and loading model..." << endmsg;

    json_config = loadJsonFile(json_path);
    
    onnx_ = std::make_unique<ONNXHelper>(model_path_clustering.value(), json_config);

   
    return StatusCode::SUCCESS;
  }

  private:
  
  nlohmann::json json_config;
  std::unique_ptr<ONNXHelper> onnx_;
  rv::RVec<std::string> vars;

  Gaudi::Property<std::string> model_path_clustering{
    this, "model_path_clustering", "/eos/user/l/lherrman/FCC/models/clustering_1.onnx",
    "Path to the ONNX clustering model"};
  
  
  Gaudi::Property<std::string> json_path{
    this, "json_path",
    "/afs/cern.ch/work/l/lherrman/private/inference/k4PFHitML/scripts/config_hits_track_v2_noise.json",
    "Path to the JSON configuration file for the ONNX model"};
  
  
};


DECLARE_COMPONENT(PFHitML)