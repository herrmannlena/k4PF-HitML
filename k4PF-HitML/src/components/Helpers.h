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
 #ifndef HELPERS_H
 #define HELPERS_H
 
 #include <fstream>
 #include <iostream>
 #include <nlohmann/json.hpp> // Include a JSON parsing library
 #include <string>
 #include <unordered_map>
 
 #include "ROOT/RVec.hxx"
 #include "Structs.h"
 

 
 /**
  * Load a JSON file from a given path.
  * @param json_path: the path to the JSON file
  * @return: the JSON object
  */
 nlohmann::json loadJsonFile(const std::string& json_path);
 

 
 #endif // HELPERS_H