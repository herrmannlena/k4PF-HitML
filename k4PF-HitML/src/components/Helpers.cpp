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

 
 std::vector<std::string>  extract_input_names(const std::string& json_path) {
   
   std::ifstream json_file(json_path);
   if (!json_file.is_open()) {
     std::cerr << "Failed to open JSON file: " << json_path << std::endl;
   }
   nlohmann::json json_config;
   json_file >> json_config;

   // Map: input name -> vector of variable names
   std::map<std::string, std::vector<std::string>> inputVarNames;

   const auto& inputs = json_config.at("inputs");
   std::vector<std::string> vars;
   for (auto& [inputName, inputParams] : inputs.items()) {
       

       // Each element of "vars" is an array like ["hit_x", null]
       for (const auto& v : inputParams.at("vars")) {
           if (v.is_array() && !v.empty() && v[0].is_string()) {
               vars.push_back(v[0].get<std::string>());
           }
       }

       inputVarNames[inputName] = vars;
   }

   
   return vars;
 }


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


    


 
 