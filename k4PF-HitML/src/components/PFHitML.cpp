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
 #include "edm4hep/ParticleIDCollection.h"
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
#include "ShowerBuilder.h"
#include "Shower.h"
#include "PFParticleBuilder.h"
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
      std::tuple<
      edm4hep::ReconstructedParticleCollection,
      edm4hep::ParticleIDCollection
      >(

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
      std::tuple<
      edm4hep::ReconstructedParticleCollection,
      edm4hep::ParticleIDCollection
      > (
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
          {
            KeyValues("PFParticles", {"PFParticles"}),
            KeyValues("PFParticleIDs", {"PFParticleIDs"})}  // Outputs
        ) {}

  // main
  std::tuple<
  edm4hep::ReconstructedParticleCollection,
  edm4hep::ParticleIDCollection
  > operator()(
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
    auto inputs_features = inputs.features;
    //convert inputs to expected shape 
    auto clustering_input = extractor.convertModelInputs(inputs_features);

    //debugging
    info() << "Prepared " << clustering_input.inputs.size()
       << " ONNX input tensor(s) for batch size "
       << clustering_input.batch_size << endmsg;

    for (size_t i = 0; i < clustering_input.inputs.size(); ++i) {
        const auto& inp = clustering_input.inputs[i];

        std::ostringstream shape_msg;
        shape_msg << "Input tensor " << i
                  << " name=" << inp.name
                  << " shape=[";

        for (size_t j = 0; j < inp.shape.size(); ++j) {
            shape_msg << inp.shape[j];
            if (j + 1 != inp.shape.size()) {
                shape_msg << ", ";
            }
        }

        shape_msg << "] type="
                  << (inp.type == ONNXInput::Type::Float ? "float" : "int64");

        if (inp.type == ONNXInput::Type::Float) {
            shape_msg << " values=" << inp.float_data.size();
        } else {
            shape_msg << " values=" << inp.int64_data.size();
        }

        info() << shape_msg.str() << endmsg;
    }



    ///////////////////////////////////////////////////
    ////////// Inference Clustering Model //////////
    ///////////////////////////////////////////////////

    
    std::vector<std::vector<float>>  outputs = m_onnx->runNamed(clustering_input.inputs);

    //std::cout << "output" << outputs[0][0] <<"" << outputs[0][1]  << "size" << outputs[0].size()<<std::endl;
    //get two outputs, first one has shape (N,4) (three coordinates in embedding space + beta)
    // second output is  dummy for pred_energy_corr (not needed at this stage)
    

    /////////////////////////////////////
    ////////// CLUSTERING STEP //////////
    /////////////////////////////////////

    // expect output tensor of length N with cluster labels


    Clustering clusterer(0.1f, 0.05f, 0.4f, 0.5f);
    torch::Tensor cluster_label = clusterer.get_clustering(outputs[0], inputs_features["node_energy"]); // length of hits [000 3 7 7 7 7 10 2 2 2 ..]

    //cluster postprocessing
    // error now in bad tracks from cluster, input feature as one hot?
    torch::Tensor cluster_label_corrected = clusterer.remove_bad_tracks_from_cluster(cluster_label, inputs_features["hit_type"], inputs_features["node_energy"], inputs_features["node_p"]);

    auto n_changed = (cluster_label != cluster_label_corrected).sum().item<int64_t>();
    std::cout << "cluster output shape: " << cluster_label_corrected.sizes() << std::endl;
    std::cout << "changed labels: " << n_changed << std::endl;

    //final output collection
    auto MLPF = edm4hep::ReconstructedParticleCollection();


 
    //the pipeline:
    //after clustering, form graphs of hits belonging to one cluster

    //this function gets the clusters, within this function create shower instances, set x,y,z,..
    ShowerBuilder builder(extractor, inputs);
    auto showers = builder.buildShowers(cluster_label_corrected, outputs[0]);
    std::cout << "shower output" << showers.size() <<std::endl;

    auto split = splitShowersByTrackContent(showers);

    info() << "Number of charged showers: " << split.charged.size() << endmsg;
    info() << "Number of neutral showers: " << split.neutral.size() << endmsg;

    

    ////////////////////////////////////
    //// ENERGY REGRESSION & PID ///////
    ////////////////////////////////////

     

    
    //look here: https://github.com/selvaggi/mlpf/blob/main/src/utils/post_clustering_features.py
    auto prop_inputs = extractor.prepare_prop(showers); //determine and convert inputs for regression model

    
    //STILL NEED TO SEPERATE NEUTRAL CHARGED! THIS IS ALSO AN OLD MODEL.. NOW I have two sepertae ones so convert them first
    // before further updating stuff..

    //////////////////////////////////////////////////
    ////////// Inference Property Model //////////////
    //////////////////////////////////////////////////

  const std::vector<int> chargedClassMap = {0, 1, 4};
  const std::vector<int> neutralClassMap = {2, 3};

  auto pfParticles = edm4hep::ReconstructedParticleCollection{};
  auto pfParticleIDs = edm4hep::ParticleIDCollection{};

  // loop over showers per event
    for (size_t idx : split.charged) {
    auto prop_outputs = m_onnx_prop_charged->runNamed(prop_inputs[idx].inputs);
    const auto& pidLogits = findPIDOutput(*m_onnx_prop_charged, prop_outputs);
    PIDPrediction pid = decodePIDLogits(pidLogits, chargedClassMap);

    ParticleRecoInfo recoInfo = buildChargedRecoInfo(showers[idx], pid.physicsClass, pid.score);

    fillRecoParticle(pfParticles, pfParticleIDs, showers[idx], recoInfo);
    
  }

  
  for (size_t idx : split.neutral) {
    auto prop_outputs = m_onnx_prop_neutral->runNamed(prop_inputs[idx].inputs);
    const auto& pidLogits = findPIDOutput(*m_onnx_prop_neutral, prop_outputs);
    PIDPrediction pid = decodePIDLogits(pidLogits, neutralClassMap);
    float predictedEnergy = prop_outputs[0].at(0);
    edm4hep::Vector3f predictedReferencePoint = computeNeutralReferencePoint(showers[idx]);
    edm4hep::Vector3f predictedDirection = computeNeutralDirection(predictedReferencePoint);


    ParticleRecoInfo recoInfo = buildNeutralRecoInfo(showers[idx], pid.physicsClass, pid.score, predictedEnergy, predictedDirection, predictedReferencePoint);

    fillRecoParticle(pfParticles, pfParticleIDs, showers[idx], recoInfo);

  }
    


  return {std::move(pfParticles), std::move(pfParticleIDs)};  //outputs

  }

  StatusCode initialize() override {
    info() << "Initializing PFHitML and loading model..." << endmsg;
    
    //Create the onnx 
    //fix input names, can you get rid of it?
    m_onnx = std::make_unique<ONNXHelper>(model_path_clustering.value());

    m_onnx_prop_neutral = std::make_unique<ONNXHelper>(model_path_properties_neutral.value());

    m_onnx_prop_charged = std::make_unique<ONNXHelper>(model_path_properties_charged.value());


    

   
    return StatusCode::SUCCESS;
  }




  private:
  

  std::unique_ptr<ONNXHelper> m_onnx;
  std::unique_ptr<ONNXHelper> m_onnx_prop_neutral;
  std::unique_ptr<ONNXHelper> m_onnx_prop_charged;
  rv::RVec<std::string> vars;

  Gaudi::Property<std::string> model_path_clustering{
    this, "model_path_clustering", "/eos/user/l/lherrman/FCC/models/clustering_truth_update.onnx",
    "Path to the ONNX clustering model"};

  Gaudi::Property<std::string> model_path_properties_neutral{
    this, "model_path_properties_neutral", "/eos/user/l/lherrman/FCC/models/energy_correction_paper_neutral.onnx",
    "Path to the ONNX model for neutral energy regression and PID"};

  Gaudi::Property<std::string> model_path_properties_charged{
    this, "model_path_properties_charged", "/eos/user/l/lherrman/FCC/models/energy_correction_paper_charged_pid.onnx",
    "Path to the ONNX model for charged energy regression and PID"};
  





};


DECLARE_COMPONENT(PFHitML)

//cleanup? get rid of helper json thing? check if this with input name in onnx part works ous