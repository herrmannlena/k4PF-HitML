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
 #include <torch/torch.h>
 #include "edm4hep/ReconstructedParticleCollection.h"
 #include "Shower.h"
 //#include "Structs.h"
 

 

  float mean_var(std::vector<float> features);
  float disperion(Shower shower_i, std::vector<edm4hep::CalorimeterHit>);
  float calculate_eta(float x, float y, float z);
  float calculate_phi(float x, float y);
  std::vector<std::vector<float>> one_hot_encode(std::vector<int>, int n_classes);
  

 
 #endif // HELPERS_H