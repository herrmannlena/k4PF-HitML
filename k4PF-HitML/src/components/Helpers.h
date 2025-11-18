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

 /*get input variables from json*/

 #ifndef HELPERS_H
 #define HELPERS_H
 
 #include <fstream>
 #include <iostream>
 #include <nlohmann/json.hpp> // Include a JSON parsing library
 #include <string>
 #include <unordered_map>
 
 #include "ROOT/RVec.hxx"
 //#include "Structs.h"
 

 
 /**
  * Load a JSON file from a given path.
  * @param json_path: the path to the JSON file
  * @return: the JSON object
  */
  std::vector<std::string> extract_input_names(const std::string& json_path);


  float energy_sys(int hit_type, std::map<std::string, std::vector<float>> features, bool squared);
  float mean(std::map<std::string, std::vector<float>> features, std::string variable);
  std::array<float,3>  mean_pos(std::map<std::string, std::vector<float>> features);
  float disperion(int hit_type, std::map<std::string, std::vector<float>> features, float n_sys_hits);

 
 #endif // HELPERS_H