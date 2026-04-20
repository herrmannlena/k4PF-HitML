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
#include "Shower.h"

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
        {"ECAL_ENDCAP", &ecalendcap_},
        {"HCAL_BARREL", &hcalbarrel_},
        {"HCAL_ENDCAP", &hcalendcap_},
        {"HCAL_OTHER",  &hcalother_},
        {"MUON",        &muons_}
    };

    
    int globalHitIndex = 0;
    int collectionIndex = 0;

    for (const auto& [name, hit_collection] : hit_collections){
        int hitIndex = 0;
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

            //include mapping
            hit_mapping.push_back({htype, collectionIndex, hitIndex});

            globalHitIndex += 1;
            hitIndex += 1;
            
        }

        collectionIndex += 1;
    }


    //extract track information
    int trackIndex = 0;
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

        hit_mapping.push_back({htype_c, collectionIndex, trackIndex});

        globalHitIndex += 1;
        trackIndex += 1;
        
    }

    features["node_energy"] = features["e_hits"];
    features["node_energy"].insert(features["node_energy"].end(),features["e_tracks"].begin(),features["e_tracks"].end());

    features["node_p"] = features["p_hits"];
    features["node_p"].insert(features["node_p"].end(),features["p_tracks"].begin(),features["p_tracks"].end());

    features["hit_type"] = features["hit_type_feature_hit"];
    features["hit_type"].insert(features["hit_type"].end(),features["hit_type_feature_track"].begin(),features["hit_type_feature_track"].end());
      
    return out;
  }


  //this is not the correct type for ONNX helper.. try alternative
  std::tuple<ONNXHelper::Tensor<float>,ONNXHelper::Tensor<long>,unsigned long long>
 DataPreprocessing::convertModelInputs(std::map<std::string, std::vector<float>> features) const {


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
    
    torch::Tensor node_e = torch::from_blob(
        const_cast<float*>(features.at("node_energy").data()),
        {static_cast<long>(features.at("node_energy").size())},
        torch::kFloat32
    ).clone().unsqueeze(1);

    torch::Tensor node_p = torch::from_blob(
        const_cast<float*>(features.at("node_p").data()),
        {static_cast<long>(features.at("node_p").size())},
        torch::kFloat32
    ).clone().unsqueeze(1);

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

    torch::Tensor hit_type = torch::from_blob(
        const_cast<float*>(features.at("hit_type").data()),
        {static_cast<long>(features.at("hit_type").size())},
        torch::kFloat32
    ).clone().unsqueeze(1);
    
   
 

    //std::cout << "pos_feature" << pos_feature.sizes()<< std::endl;
    //std::cout << "type_feature" << hit_type_feature.sizes()<< std::endl;
    //std::cout << "p_feature" << p_feature.sizes()<< std::endl;
    //std::cout << "e_feature" << e_feature.sizes()<< std::endl;
    //std::cout << "onehot" << hit_type_one_hot.sizes() << std::endl;

    //final onnx input
    torch::Tensor h = torch::cat({pos_feature, hit_type, node_e, node_p}, 1).to(torch::kFloat32);

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

  

  

  //prepare the inputs for energy regression and PID, return node and global features
  std::vector<ModelInputs> DataPreprocessing::prepare_prop(std::vector<Shower> showers) const {
    
    //loop over showers   
    std::vector<ModelInputs> out;
    out.reserve(showers.size());

    for (auto& shower_i : showers) {

        ONNXHelper::Tensor<float> node_features;
        node_features.reserve(9);
    
        std::vector<float> global_features;
        global_features.reserve(16);
        //these are used as inputs. Find order
        //also fakes? do I need to also implement that part?

        //variab;es that are fed to gatr, investigate what to do

        //pos from calo and track
        auto [pos_x, pos_y, pos_z] = shower_i.get_pos();

        std::vector<std::vector<float>> hit_one_hot = one_hot_encode(shower_i.types_, 4); //3-6

        auto [e_vector, p_vector] = shower_i.get_ep();

        std::vector<float> betas = shower_i.betas_; //9

        node_features.push_back(pos_x);
        node_features.push_back(pos_y);
        node_features.push_back(pos_z);
        // hit_one_hot: [n_hits][4]
        for (size_t k = 0; k < 4; ++k) {
            std::vector<float> one_hot_feature;
            one_hot_feature.reserve(hit_one_hot.size());

            for (size_t i = 0; i < hit_one_hot.size(); ++i) {
                one_hot_feature.push_back(hit_one_hot[i][k]);
            }

            node_features.push_back(one_hot_feature);  // 3,4,5,6
        }
        node_features.push_back(e_vector);
        node_features.push_back(p_vector);
        node_features.push_back(betas);

        

        //include distinction charged neutral..
       

        // high level stuff

        float sum_e = shower_i.getCaloEnergy(shower_i.caloHits_).first; 
        float muon_e = shower_i.getCaloEnergy(shower_i.muonHits_).first; 
        float ecal_e = shower_i.getCaloEnergy(shower_i.ecalHits_).first;
        float hcal_e = shower_i.getCaloEnergy(shower_i.hcalHits_).first;

        float ECAL_e_fraction = ecal_e / sum_e; 
        float HCAL_e_fraction = hcal_e / sum_e; 

        int num_hits = shower_i.getCalorimeterHits().size(); 
        int num_tracks = shower_i.getTracks().size(); 
        int num_muon_hits = shower_i.muonHits_.size(); 

        int total_hits = num_hits + num_tracks;   // include tracks or not? 

        float track_p = shower_i.getTrackMomentum_mean(); 

        float dispersion_ecal = disperion(shower_i, shower_i.ecalHits_); 
        float dispersion_hcal = disperion(shower_i, shower_i.hcalHits_); 

        float chi2 = std::clamp(shower_i.Chi2_mean(), -5.0f, 5.0f); 

        float mean_x = mean_var(pos_x); 
        float mean_y = mean_var(pos_y); 
        float mean_z = mean_var(pos_z); 

        float eta = calculate_eta(mean_x, mean_y, mean_z); 
        float phi = calculate_phi(mean_x, mean_y); 

        //add the global features, take right order!
        

        global_features.push_back(ECAL_e_fraction); //0
        global_features.push_back(HCAL_e_fraction); //1
        global_features.push_back(static_cast<float>(num_hits)); //2
        global_features.push_back(track_p); //3
        global_features.push_back(dispersion_ecal); //4
        global_features.push_back(dispersion_hcal); //5
        global_features.push_back(sum_e); //6
        global_features.push_back(static_cast<float>(num_tracks)); //7
        global_features.push_back(chi2); //8
        global_features.push_back(muon_e); //9
        global_features.push_back(static_cast<float>(num_muon_hits)); //10
        global_features.push_back(mean_x); //11
        global_features.push_back(mean_y); //12
        global_features.push_back(mean_z); //13
        global_features.push_back(eta); //14
        global_features.push_back(phi); //15

        ONNXHelper::Tensor<float> g_tensor;
        g_tensor.reserve(1);
        g_tensor.push_back(global_features);
      
        

        //pos is a track thing?

        //check if the calculation of these quantities correct? just mean? i.e. scatter sum? 
        //some other transformations?

        //right order?
        out.emplace_back(std::move(node_features), std::move(g_tensor));

        

    }



    return out;
  }


  // mache eine funktion, die vorbereitet fuer model Format
//sind die track states correct?
//hit type correct?, shouldn't it be N,1?