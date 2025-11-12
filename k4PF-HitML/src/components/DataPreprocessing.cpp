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

 Description: 
 converts rec files into the required format for MLPF inference. The standalone repository doing this job can be found here: https://github.com/doloresgarcia/MLPF_datageneration
 */

#include "DataPreprocessing.h"

//edm4hep imports
#include "edm4hep/TrackerHit.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"

DataPreprocessing::DataPreprocessing(
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
  

std::map<std::string, std::vector<float>> DataPreprocessing::extract() const {
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
            auto energy = hit.getEnergy();
            
            float x = pos.x;
            float y = pos.y;
            float z = pos.z;
            
            int htype;
            if (name.find("ECAL") != std::string::npos) {
                htype = 2;
            } 
            else if (name.find("HCAL") != std::string::npos) {
                htype = 3;
            } else if (name.find("MUON") != std::string::npos) {
                htype = 4;
            }
            else{
                htype = 5; 
            }


            features["pos_hits_xyz_hits"].push_back(x);
            features["pos_hits_xyz_hits"].push_back(y);
            features["pos_hits_xyz_hits"].push_back(z);
            features["e_hits"].push_back(energy);
            features["p_hits"].push_back(0);
            features["hit_type_feature_hit"].push_back(htype);
            
        }
    }

    //extract track information
    for (const auto& track : tracks_) {
        
        //trackstate at IP?
        auto trackstate = track.getTrackStates()[1];
        float omega = trackstate.omega;
        float phi = trackstate.phi;
        float tanLambda = trackstate.tanLambda;

        float pt = 2.99792e-4 * std::abs(2.0/omega);  // B filed 2T
        float px = std::cos(phi) * pt;
        float py = std::sin(phi) * pt;
        float pz = tanLambda  * pt;
        float p = std::sqrt(px * px + py * py + pz * pz);
   
        features["p_tracks"].push_back(p);
        features["e_tracks"].push_back(0);  

        //also add features for trackstate at calo
        auto trackstate_calo = track.getTrackStates()[3];
        auto referencePoint_calo = trackstate_calo.referencePoint;

        float x_c = referencePoint_calo.x;
        float y_c = referencePoint_calo.y;
        float z_c = referencePoint_calo.z;

        int htype_c = 1; //vertex track state
    
        features["pos_hits_xyz_tracks"].push_back(x_c);
        features["pos_hits_xyz_tracks"].push_back(y_c);
        features["pos_hits_xyz_tracks"].push_back(z_c);
        features["hit_type_feature_track"].push_back(htype_c);
        
    }
      
    return features;
  }


  // mache eine funktion, die vorbereitet fuer model Format
//sind die track states correct?