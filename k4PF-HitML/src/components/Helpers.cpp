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




    


 
 