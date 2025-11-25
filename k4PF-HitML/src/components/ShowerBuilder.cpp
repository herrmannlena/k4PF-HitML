#include "ShowerBuilder.h"
#include "Shower.h"
#include "DataPreprocessing.h"

std::vector<Shower> ShowerBuilder::buildShowers(const torch::Tensor& cluster_label){

    torch::Tensor uniqueTensor; // enumerates showers [0,1,2,3]
    torch::Tensor inverseIndices; // each hit gets shower label i.e. [3,3,3,3,2,3,0,0,0,2,2,2,2,1,1,1,1]
    std::tie(uniqueTensor, inverseIndices) = at::_unique(cluster_label, true, true);
  
  
    //number of particles
    int64_t num_part = uniqueTensor.numel();
  
    auto uniqueView = uniqueTensor.accessor<int64_t, 1>();
  
    std::cout << "num particles" << num_part <<std::endl;
    std::vector<Shower> showers(num_part);
   
    for (int64_t i = 0; i < num_part; ++i) {
  
      int64_t label = uniqueView[i];
  
      // Create a PF shower object
      //auto pf = MLPF.create();
  
      // Mask hits belonging to this cluster
      torch::Tensor mask    = (cluster_label == label);           // [00011000]
      torch::Tensor indices = torch::nonzero(mask).flatten();     // [3,4]
  
  
      auto idxView = indices.accessor<int64_t, 1>();
      const auto& hit_mapping = preproc_.hit_mapping;

      
  
      // assign the hits to the shower object.
      //understand which properties you can add like mean x of clusters..
      // I think I would fill all of this in a shower class and in the end if all values are fixed and assigned pass and fill to MLPF object
  
     
      for (int64_t j = 0; j < indices.size(0); ++j) {
  
          int64_t hitIdxModel = idxView[j];       // index in NN input order
  
          int64_t htype = hit_mapping[hitIdxModel][0];
          int64_t coll  = hit_mapping[hitIdxModel][1];
          int64_t hidx  = hit_mapping[hitIdxModel][2];
          
          //shower object
          Shower& shower_i = showers[i];

          edm4hep::CalorimeterHit hit_i;
          edm4hep::Track track_i;
          
          //assign tracks
          if (htype == 1){
            track_i = dp_.tracks().at(hidx);
          }
          else if (htype == 2){
            if(coll == 0){
                hit_i = dp_.ecalBarrel().at(hidx);
            }
            if(coll == 2){
                hit_i = dp_.ecalEndcap().at(hidx);
            }
          }
          else if (htype == 3){
            if(coll == 1){
                hit_i = dp_.hcalBarrel().at(hidx);
            }
            else if(coll == 3){
                hit_i = dp_.hcalEndcap().at(hidx);
            }
            else if(coll == 4){
                hit_i = dp_.hcalOther().at(hidx);
            }
          }
          else if (htype == 4){
            hit_i = dp_.muons().at(hidx);
          }
          


          if(hit_i.isAvailable()){
            shower_i.addCalorimeterHit(hit_i);
          }

          if(track_i.isAvailable()){
            shower_i.addTrack(track_i);
          }
             
  
         
  
      }
          
         
  
    }
      
  
  
    return showers;
  }
  
  