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


  // var = E(x^2) - E(x)^2
 float disperion(int hit_type, std::map<std::string, std::vector<float>> features, float n_sys_hits){

    float energy_sum = energy_sys(hit_type, features, false);
    float energy_sum_sq = energy_sys(hit_type, features, true);

    float dispersion = energy_sum_sq / n_sys_hits - std::pow((energy_sum/ n_sys_hits),2);
    
    return dispersion;
 }


    


 
 