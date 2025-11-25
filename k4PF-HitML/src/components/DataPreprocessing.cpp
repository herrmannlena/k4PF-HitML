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

#include "Helpers.h"

#include <torch/torch.h>
#include "ONNXHelper.h"  

DataPreprocessing::DataPreprocessing(
    const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalOther_hits,
    const edm4hep::CalorimeterHitCollection& Muon_hits,
    const edm4hep::TrackCollection& tracks)
    : ecalbarrel_(EcalBarrel_hits), hcalbarrel_(HcalBarrel_hits),
    ecalendcap_(EcalEndcap_hits), hcalendcap_(HcalEndcap_hits), hcalother_(HcalOther_hits), 
    muons_(Muon_hits), tracks_(tracks){}


    
  

PreprocessedData DataPreprocessing::extract() const {
    //std::map<std::string, std::vector<float>> features; //features used for clustering model
    PreprocessedData out;
    auto& features = out.features;
    auto& hit_mapping = out.hit_mapping;

    // collection of hits
    std::vector<std::pair<std::string, const edm4hep::CalorimeterHitCollection*>> hit_collections = {
        {"ECAL_BARREL", &ecalbarrel_},
        {"HCAL_BARREL", &hcalbarrel_},
        {"ECAL_ENDCAP", &ecalendcap_},
        {"HCAL_ENDCAP", &hcalendcap_},
        {"HCAL_OTHER",  &hcalother_},
        {"MUON",        &muons_}
    };

    
    float hit_e_sum = 0;
    int globalHitIndex = 0;
    int collectionIndex = 0;

    for (const auto& [name, hit_collection] : hit_collections){
        int hitIndex = 0;
        for (const auto& hit : *hit_collection){

            auto pos = hit.getPosition();
            auto energy = hit.getEnergy();
            hit_e_sum += energy;
            
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

            //include mapping
            hit_mapping.push_back({htype, collectionIndex, hitIndex});

            globalHitIndex += 1;
            hitIndex += 1;
            
        }

        collectionIndex += 1;
    }

    features["hit_e_sum"].push_back(hit_e_sum);

    //extract track information
    int trackIndex = 0;
    for (const auto& track : tracks_) {
        
        //trackstate at IP?
        auto trackstate = track.getTrackStates()[1];
        float omega = trackstate.omega;
        float phi = trackstate.phi;
        float tanLambda = trackstate.tanLambda;
        float chi2 = track.getChi2();

        float pt = 2.99792e-4 * std::abs(2.0/omega);  // B filed 2T
        float px = std::cos(phi) * pt;
        float py = std::sin(phi) * pt;
        float pz = tanLambda  * pt;
        float p = std::sqrt(px * px + py * py + pz * pz);
   
        features["p_tracks"].push_back(p);
        features["e_tracks"].push_back(0);  
        features["chi2"].push_back(chi2); 

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

        hit_mapping.push_back({htype_c, collectionIndex, trackIndex});

        globalHitIndex += 1;
        trackIndex += 1;
        
    }
      
    return out;
  }


  //this is not the correct type for ONNX helper.. try alternative
  std::tuple<ONNXHelper::Tensor<float>,ONNXHelper::Tensor<long>,unsigned long long>
 DataPreprocessing::convertModelInputs(std::map<std::string, std::vector<float>> features) const {

    // prepare hit type
    torch::Tensor hit_type   = torch::from_blob(
        const_cast<float*>(features.at("hit_type_feature_hit").data()),          // pointer to data
        {static_cast<long>(features.at("hit_type_feature_hit").size())},         // shape
        torch::kFloat32                                                         // dtype
    ).clone(); 
    
    torch::Tensor track_type = torch::from_blob(
        const_cast<float*>(features.at("hit_type_feature_track").data()),
        {static_cast<long>(features.at("hit_type_feature_track").size())},
        torch::kFloat32
    ).clone();
    
    //concatenate hit type features
    torch::Tensor hit_type_feature = torch::cat({hit_type, track_type}, 0); 
    hit_type_feature = hit_type_feature.unsqueeze(1);

    //one hot
    //torch::Tensor hit_type_one_hot = torch::one_hot(
    //    hit_type_feature.to(torch::kInt64), // input tensor of integer class indices
    //    /* num_classes = */ 5
    //).to(torch::kFloat32); 
    

    // prepare position 
    const auto& pos_hits_flat = features.at("pos_hits_xyz_hits");  
    std::size_t N = pos_hits_flat.size() / 3;

    torch::Tensor pos_hits = torch::tensor(pos_hits_flat, torch::kFloat32).reshape({static_cast<long>(N), 3});

    const auto& pos_tracks_flat = features.at("pos_hits_xyz_tracks");  
    std::size_t N_tracks = pos_tracks_flat.size() / 3;
                           
    torch::Tensor pos_tracks = torch::tensor(pos_tracks_flat, torch::kFloat32).reshape({static_cast<long>(N_tracks), 3});
  
    //concatenate pos features
    torch::Tensor pos_feature = torch::cat({pos_hits, pos_tracks}, 0); 

    //prepare e 
    torch::Tensor hit_e   = torch::from_blob(
        const_cast<float*>(features.at("e_hits").data()),          
        {static_cast<long>(features.at("e_hits").size())},         
        torch::kFloat32                                                        
    ).clone(); 
    
    torch::Tensor track_e = torch::from_blob(
        const_cast<float*>(features.at("e_tracks").data()),
        {static_cast<long>(features.at("e_tracks").size())},
        torch::kFloat32
    ).clone();
    
    //concatenate 
    torch::Tensor e_feature = torch::cat({hit_e, track_e}, 0); 
    e_feature = e_feature.unsqueeze(1);

    //prepare p
    torch::Tensor hit_p   = torch::from_blob(
        const_cast<float*>(features.at("p_hits").data()),          
        {static_cast<long>(features.at("p_hits").size())},         
        torch::kFloat32                                                        
    ).clone(); 
    
    torch::Tensor track_p = torch::from_blob(
        const_cast<float*>(features.at("p_tracks").data()),
        {static_cast<long>(features.at("p_tracks").size())},
        torch::kFloat32
    ).clone();
    
    //concatenate 
    torch::Tensor p_feature = torch::cat({hit_p, track_p}, 0); 
    p_feature = p_feature.unsqueeze(1);
 

    //std::cout << "pos_feature" << pos_feature.sizes()<< std::endl;
    //std::cout << "type_feature" << hit_type_feature.sizes()<< std::endl;
    //std::cout << "p_feature" << p_feature.sizes()<< std::endl;
    //std::cout << "e_feature" << e_feature.sizes()<< std::endl;
    //std::cout << "onehot" << hit_type_one_hot.sizes() << std::endl;

    //final onnx input
    torch::Tensor h = torch::cat({pos_feature, hit_type_feature, e_feature, p_feature}, 1).to(torch::kFloat32);

    //convert for ONNXHelper
    size_t numel = static_cast<size_t>(h.numel());
    std::vector<float> flat(numel);
    std::memcpy(flat.data(), h.data_ptr<float>(), numel * sizeof(float));
    ONNXHelper::Tensor<float> input_tensor;
    input_tensor.emplace_back(std::move(flat));


    ONNXHelper::Tensor<long> input_shapes;
    input_shapes.emplace_back();
    input_shapes.back().push_back(h.size(0));
    input_shapes.back().push_back(h.size(1));


    //m_inputShapes = { std::vector<long>(h.sizes().begin(), h.sizes().end()) };
     
    //std::cout << "m_inputShapes" << m_inputShapes[0][0] << std::endl;
    //std::cout << "m_inputShapes" << m_inputShapes[0][1] << std::endl;
  
    return {input_tensor, input_shapes, h.size(0)};

  }

  
  //prepare the inputs for energy regression and PID
  //I think this is not correct. features per physical cluster?? check this again
  std::vector<float> DataPreprocessing::prepare_prop(std::map<std::string, std::vector<float>> features) const {
    
    //std::map<std::string, std::vector<float>> features_prop; //features for property determination
    int num_hits = features.at("hit_type_feature_hit").size(); 
    int num_tracks = tracks_.size(); 

    
    //these are used as inputs. Find order
    float sum_e = features.at("hit_e_sum")[0];
    float ECAL_e_fraction = energy_sys(2, features, false) / sum_e;
    float HCAL_e_fraction = energy_sys(3, features, false) / sum_e;
    float Muon_e = energy_sys(4, features, false);
    float num_muon = muons_.size(); 
    float track_p = mean(features, "p_tracks"); //take mean momentum of tracks
    float n_ecal_hits = ecalbarrel_.size() + ecalendcap_.size();
    float n_hcal_hits = hcalbarrel_.size() + hcalendcap_.size() + hcalother_.size();
    float dispersion_ecal = disperion(2, features, n_ecal_hits);
    float dispersion_hcal = disperion(3, features, n_hcal_hits);
    float chi2 = std::clamp(mean(features, "chi2"), -5.0f, 5.0f); 

    std::array<float,3> mean_xyz = mean_pos(features);

    float mean_x = mean_xyz[0];
    float mean_y = mean_xyz[1];
    float mean_z = mean_xyz[2];
    float eta = calculate_eta(mean_x, mean_y, mean_z);
    float phi = calculate_phi(mean_x, mean_y);





    




    return features.at("p_hits");
  }


  // mache eine funktion, die vorbereitet fuer model Format
//sind die track states correct?
//hit type correct?, shouldn't it be N,1?
//eta, phi und mean x,y,z is do iwie das selbe