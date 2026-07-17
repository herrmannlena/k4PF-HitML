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
#ifndef SHOWERBUILDER_H
#define SHOWERBUILDER_H

#include "DataPreprocessing.h"
#include "Shower.h"

//edm4hep imports
#include "edm4hep/TrackerHit.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"



class ShowerBuilder {
    public:
        ShowerBuilder(

            const DataPreprocessing& dp,
            const PreprocessedData& preproc)
            : dp_(dp), preproc_(preproc)

        {}

        std::vector<Shower> buildShowers(const torch::Tensor& cluster_label, const std::vector<float>& betas);

    private:
        const DataPreprocessing& dp_;
        const PreprocessedData& preproc_;
    };

#endif // SHOWERHELPER_H
