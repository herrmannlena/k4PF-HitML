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
    const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalOther_hits,
    const edm4hep::CalorimeterHitCollection& Muon_hits,
    const edm4hep::TrackCollection& tracks)
    : mc_(mc_particles), ecalbarrel_(EcalBarrel_hits), hcalbarrel_(HcalBarrel_hits),
    ecalendcap_(EcalEndcap_hits), hcalendcap_(HcalEndcap_hits), hcalother_(HcalOther_hits), 
    muons_(Muon_hits), tracks_(tracks){}
  

std::map<std::string, std::vector<float>> ObservableExtractor::extract() const {
    std::map<std::string, std::vector<float>> features;

    // collection of hits
    std::vector<std::pair<std::string, const edm4hep::CalorimeterHitCollection*>> hit_collections = {
        {"ECAL_BARREL", &ecalbarrel_},
        {"HCAL_BARREL", &hcalbarrel_},
        {"ECAL_ENDCAP", &ecalendcap_},
        {"HCAL_ENDCAP", &hcalendcap_},
        {"HCAL_OTHER",  &hcalother_},
        {"MUON",        &muons_}
    };


    for (const auto& [name, hit_collection] : hit_collections){
        for (const auto& hit : *hit_collection){

            auto pos = hit.getPosition();
            auto t = hit.getTime();
            auto energy = hit.getEnergy();
            
            float x = pos.x;
            float y = pos.y;
            float z = pos.z;
            //float r = std::sqrt(x*x+y*y+z*z);
            //float theta = std::acos(z/r);
            //float phi = std::atan2(y, x);

            int htype = 2;  // Default to 2 (ECAL)
            if (name.find("HCAL") != std::string::npos) {
                htype = 3;
            } else if (name.find("MUON") != std::string::npos) {
                htype = 4;
            }

            features["hit_x"].push_back(x);
            features["hit_y"].push_back(y);
            features["hit_z"].push_back(z);
            features["hit_t"].push_back(t);
            features["hit_e"].push_back(energy);
            //features["hit_theta"].push_back(theta);
            //features["hit_phi"].push_back(phi);
            features["hit_htype"].push_back(htype);
            
        }
    }

    //extract track information
    for (const auto& track : tracks_) {

        auto trackstate = track.getTrackStates()[0];
        auto referencePoint = trackstate.referencePoint;
        float x = referencePoint.x;
        float y = referencePoint.y;
        float z = referencePoint.z;

        int htype_v = 0; //vertex track state
    
        features["hit_x"].push_back(x);
        features["hit_y"].push_back(y);
        features["hit_z"].push_back(z);
        features["hit_t"].push_back(trackstate.time);
        features["hit_e"].push_back(-1);  //why dummy?
        features["hit_htype"].push_back(htype_v);


        //also add features for trackstate at calo
        auto trackstate_calo = track.getTrackStates()[3];
        auto referencePoint_calo = trackstate_calo.referencePoint;

        float x_c = referencePoint_calo.x;
        float y_c = referencePoint_calo.y;
        float z_c = referencePoint_calo.z;

        int htype_c = 1; //vertex track state
    
        features["hit_x"].push_back(x_c);
        features["hit_y"].push_back(y_c);
        features["hit_z"].push_back(z_c);
        features["hit_t"].push_back(trackstate_calo.time);
        features["hit_e"].push_back(-1);  //why dummy?
        features["hit_htype"].push_back(htype_c);
        
    }
    

  
    return features;
  }