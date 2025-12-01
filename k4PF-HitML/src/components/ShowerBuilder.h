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
    