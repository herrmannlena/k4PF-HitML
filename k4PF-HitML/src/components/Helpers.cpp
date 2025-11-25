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
 #include "Helpers.h"

 #include "Gaudi/Property.h"
 #include <torch/torch.h>
 #include "edm4hep/ReconstructedParticleCollection.h"

 

 //determine energy of subsystem
 float energy_sys(int hit_type, std::map<std::string, std::vector<float>> features, bool squared){
  float e_cal = 0;
  float e_cal_sq = 0;
  
   if (features.at("e_hits").size() != features.at("hit_type_feature_hit").size()) {
      throw std::runtime_error("e_hits and hit_types size mismatch");
   }

  for (size_t i = 0; i < features.at("hit_type_feature_hit").size(); ++i) {
   
    if (features.at("hit_type_feature_hit")[i] == hit_type) {
        e_cal += features.at("e_hits")[i];   
        e_cal_sq += std::pow(features.at("e_hits")[i],2);
    }
  }

  if (squared){
    return e_cal_sq;
  }
  else{
    return e_cal ;
  }
  

 }


 float mean(std::map<std::string, std::vector<float>> features, std::string variable){
  
  float temp = 0; 
  float num_var = features.at(variable).size();

  for (size_t i = 0; i < features.at(variable).size(); ++i) {
    temp += features.at(variable)[i];   
  }

  if (num_var == 0) return 0.0f;
  return temp / num_var;

 }


 std::array<float,3> mean_pos(std::map<std::string, std::vector<float>> features){

  std::vector<float> xyz = features.at("pos_hits_xyz_tracks");
  
  float x_sum = 0.0f, y_sum = 0.0f, z_sum = 0.0f;
  int n_hits = xyz.size() / 3;
  
  for (int i = 0; i < n_hits; ++i) {
      x_sum += xyz[3*i + 0];
      y_sum += xyz[3*i + 1];
      z_sum += xyz[3*i + 2];
  }
  
  float x_mean;
  float y_mean;
  float z_mean;

  if (n_hits > 0) {
      x_mean = x_sum / n_hits;
      y_mean = y_sum / n_hits;
      z_mean = z_sum / n_hits;
  } else {
      x_mean = 0.0f;
      y_mean = 0.0f;
      z_mean = 0.0f;
  }

  return {x_mean, y_mean, z_mean};

 }


  // var = E(x^2) - E(x)^2
 float disperion(int hit_type, std::map<std::string, std::vector<float>> features, float n_sys_hits){

    float energy_sum = energy_sys(hit_type, features, false);
    float energy_sum_sq = energy_sys(hit_type, features, true);

    float dispersion = energy_sum_sq / n_sys_hits - std::pow((energy_sum/ n_sys_hits),2);
    
    return dispersion;
 }

 float calculate_phi(float x, float y) {
  // matches numpy arctan2(y, x)
  return std::atan2(y, x);
}

float calculate_eta(float x, float y, float z) {
  // theta = arctan2( sqrt(x^2 + y^2), z )
  float r = std::sqrt(x*x + y*y);
  float theta = std::atan2(r, z);

  // eta = -log( tan(theta/2) )
  float tan_half_theta = std::tan(theta / 2.0f);

  // avoid log(0) if theta is 0
  if (tan_half_theta == 0.0f)
      return 0.0f;

  return -std::log(tan_half_theta);
}

/*
float get_particle(torch::Tensor cluster_label, std::map<std::string, std::vector<float>> inputs,  edm4hep::ReconstructedParticleCollection& MLPF){
  
  torch::Tensor uniqueTensor; // enumerates showers [0,1,2,3]
  torch::Tensor inverseIndices; // each hit gets shower label i.e. [3,3,3,3,2,3,0,0,0,2,2,2,2,1,1,1,1]
  std::tie(uniqueTensor, inverseIndices) = at::_unique(cluster_label, true, true);


  //number of particles
  int64_t num_part = uniqueTensor.numel();

  auto uniqueView = uniqueTensor.accessor<int64_t, 1>();

  std::cout << "num particles" << num_part <<std::endl;
 
  for (int64_t i = 0; i < num_part; ++i) {

    int64_t label = uniqueView[i];

    // Create a PF shower object
    auto pf = MLPF.create();

    // Mask hits belonging to this cluster
    torch::Tensor mask    = (cluster_label == label);           // [00011000]
    torch::Tensor indices = torch::nonzero(mask).flatten();     // [3,4]


    auto idxView = indices.accessor<int64_t, 1>();

    // assign the hits to the shower object.
    //understand which properties you can add like mean x of clusters..
    // I think I would fill all of this in a shower class and in the end if all values are fixed and assigned pass and fill to MLPF object


    for (int64_t j = 0; j < indices.size(0); ++j) {

        int64_t hitIdxModel = idxView[j];       // index in NN input order

        int64_t htype = hitMapView[hitIdxModel][0];
        int64_t coll  = hitMapView[hitIdxModel][1];
        int64_t hidx  = hitMapView[hitIdxModel][2];

        // Now fetch the original hit from the right collection
        const edm4hep::CalorimeterHit* hitPtr = nullptr;

        if (htype == 2) {               // ECAL
            auto collPtr = ecalCollections[coll];
            hitPtr = &collPtr->at(hidx);
        } else if (htype == 3) {        // HCAL
            auto collPtr = hcalCollections[coll];
            hitPtr = &collPtr->at(hidx);
        } else if (htype == 4) {        // MUON
            auto collPtr = muonCollections[coll];
            hitPtr = &collPtr->at(hidx);
        } else {
            auto collPtr = otherCaloCollections[coll];
            hitPtr = &collPtr->at(hidx);
        }

        // Attach hit to the PF object
        // (exact name depends on your EDM4hep bindings)
        pf.addToClusters(*hitPtr);      // or addToParticleIDs, etc.

        //add hits to shower object. later function to get energy corrction properties

    }
       

  }
    


  return 1.0;
}
*/

    


 
 