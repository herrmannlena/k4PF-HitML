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

//others
#include "DataPreprocessing.h"
#include "Clustering.h"  
#include "Helpers.h"  
#include "ROOT/RVec.hxx"
#include <nlohmann/json.hpp> 

//ONNX
#include "onnxruntime_cxx_api.h"
#include <torch/torch.h>
#include "ONNXHelper.h" 

namespace rv = ROOT::VecOps;

 /**
  output: collection
  */



struct PFHitML final:
   k4FWCore::MultiTransformer<
   std::tuple<>(

    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::TrackCollection&

   ) > {

    PFHitML(const std::string& name, ISvcLocator* svcLoc)
      : k4FWCore::MultiTransformer<
      std::tuple<>(
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::TrackCollection&)>(
        name, svcLoc,
          {
            KeyValues("EcalBarrelHits", {"ECALBarrel"}),
            KeyValues("HcalBarrelHits", {"HCALBarrel"}),
            KeyValues("EcalEndcaplHits", {"ECALEndcap"}),
            KeyValues("HcalEndcapHits", {"HCALEndcap"}),
            KeyValues("HcalOtherHits", {"HCALOther"}),
            KeyValues("MUON", {"MUON"}),
            KeyValues("Tracks", {"SiTracks_Refitted"})
          },
          {}  // no Outputs
        ) {}

  // main
  std::tuple<> operator()(
    const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalOther_hits,
    const edm4hep::CalorimeterHitCollection& Muon_hits,
    const edm4hep::TrackCollection& tracks
  ) const override {

    info() << "EcalBarrelHits: " << EcalBarrel_hits.size() << endmsg;
    info() << "HcalBarrelHits: " << HcalBarrel_hits.size() << endmsg;
    info() << "EcalEndcapHits: " << EcalEndcap_hits.size() << endmsg;
    info() << "HcalEndcapHits: " << HcalEndcap_hits.size() << endmsg;
    info() << "HcalOtherHits: " << HcalOther_hits.size() << endmsg;
    info() << "MuonHits: " << Muon_hits.size() << endmsg;
    info() << "tracks: " << tracks.size() << endmsg;

    DataPreprocessing extractor(
      EcalBarrel_hits, 
      HcalBarrel_hits, 
      EcalEndcap_hits, 
      HcalEndcap_hits, 
      HcalOther_hits,
      Muon_hits,
      tracks
    );
    
    //get input variables
    auto inputs = extractor.extract();
    //convert inputs to expected shape 
    auto [inputs_onnx, input_shapes, batch_size] = extractor.convertModelInputs(inputs);


    ///////////////////////////////////////////////////
    ////////// Inference Clustering Model //////////
    ///////////////////////////////////////////////////

    
    std::vector<std::vector<float>>  outputs = m_onnx->run(inputs_onnx, input_shapes, batch_size);

    //std::cout << "output" << outputs[0][0] <<"" << outputs[0][1]  <<std::endl;
    //get two outputs, first one has shape (N,4) (three coordinates in embedding space + beta)
    // second output is  dummy for pred_energy_corr (not needed at this stage)
    

    /////////////////////////////////////
    ////////// CLUSTERING STEP //////////
    /////////////////////////////////////

    // expect output tensor of length N with cluster labels
    Clustering clusterer(0.5, 0.2);
    torch::Tensor cluster_label = clusterer.get_clustering(outputs[0]);

    std::cout << "cluster output" << cluster_label.sizes() <<std::endl;

    torch::Tensor uniqueTensor;
    torch::Tensor inverseIndices;
    std::tie(uniqueTensor, inverseIndices) = at::_unique(cluster_label, true, true);

    //final output collection
    auto MLPF = edm4hep::ReconstructedParticleCollection();

    ////////////////////////////////////
    //// ENERGY REGRESSION & PID ///////
    ////////////////////////////////////
    
    //look here: https://github.com/selvaggi/mlpf/blob/main/src/utils/post_clustering_features.py
    auto prop_inputs = extractor.prepare_prop(inputs); //determine and convert inputs for regression model
    

    
    


    return {}; // no outputs
  }

  StatusCode initialize() override {
    info() << "Initializing PFHitML and loading model..." << endmsg;
    
    //fix this.. not needed?
    input_names = extract_input_names(json_path);
    
    //Create the onnx 
    //fix input names, provide only orderd inputs..
    m_onnx = std::make_unique<ONNXHelper>(model_path_clustering.value(), names);


    

   
    return StatusCode::SUCCESS;
  }




  private:
  
  std::vector<std::string> input_names;
  //std::vector<std::string> names = {"X_hit", "X_track"};
  //names that are saved in the model
  std::vector<std::string> names = {"inputs"};

  std::unique_ptr<ONNXHelper> m_onnx;
  rv::RVec<std::string> vars;

  Gaudi::Property<std::string> model_path_clustering{
    this, "model_path_clustering", "/eos/user/l/lherrman/FCC/models/clustering_truth_update.onnx",
    "Path to the ONNX clustering model"};
  
  
  Gaudi::Property<std::string> json_path{
    this, "json_path",
    "/afs/cern.ch/work/l/lherrman/private/inference/k4PFHitML/scripts/config_hits_track_v2_noise.json",
    "Path to the JSON configuration file for the ONNX model"};




};


DECLARE_COMPONENT(PFHitML)

//cleanup? get rid of helper json thing? check if this with input name in onnx part works ous