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

 #include "k4FWCore/Transformer.h"
 #include "edm4hep/MCParticleCollection.h"
 #include <string>


//edm4hep imports
#include "edm4hep/TrackerHit.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"


 
 /**
  include description
  */





struct PFHitML final:
   k4FWCore::MultiTransformer<
   std::tuple<>(

    const edm4hep::MCParticleCollection&

   )> {

    PFHitML(const std::string& name, ISvcLocator* svcLoc)
      : k4FWCore::MultiTransformer<
      std::tuple<>(
        const edm4hep::MCParticleCollection&)>(
        name, svcLoc,
          {
            KeyValues("MCParticles", {"MCParticles"})
          },
          {}  // no Outputs
        ) {}

  // main
  std::tuple<> operator()(
    const edm4hep::MCParticleCollection& mc_particles
  ) const override {

    info() << "MCParticles: " << mc_particles.size() << endmsg;

    return {}; // no outputs
  }
};


DECLARE_COMPONENT(PFHitML)