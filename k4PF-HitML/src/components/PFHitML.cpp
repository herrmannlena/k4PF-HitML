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
#include "edm4hep/RecoMCParticleLinkCollection.h"
#include "edm4hep/CaloHitMCParticleLinkCollection.h"
#include "edm4hep/TrackMCParticleLinkCollection.h"


//others
#include "DataPreprocessing.h"
#include "Clustering.h"  
#include "Helpers.h"  
#include "ShowerBuilder.h"
#include "Shower.h"
#include "PFParticleBuilder.h"
#include "ROOT/RVec.hxx"
#include <nlohmann/json.hpp> 
#include "TruthMatcher.h"
#include <optional>
#include <unordered_map>
#include <limits>
#include <fstream>
#include <iomanip>
#include "EvaluationSummary.h"



//ONNX
#include "onnxruntime_cxx_api.h"
#include <torch/torch.h>
#include "ONNXHelper.h" 

using RecoTruthLinkCollection = edm4hep::RecoMCParticleLinkCollection;
using CaloTruthLinkCollection = edm4hep::CaloHitMCParticleLinkCollection;
using TrackTruthLinkCollection = edm4hep::TrackMCParticleLinkCollection;



namespace rv = ROOT::VecOps;

namespace {

// Validation dump: raw clustering-model inputs, one row per hit/track node,
// in the exact order they were fed to ONNX (pos_hits_xyz, hit_type, h_scalar).
void dumpClusteringInputs(const ClusteringInputs& clustering_input, std::size_t eventIdx) {
    const auto& pos = clustering_input.inputs.at(0).float_data;       // pos_hits_xyz [N,3]
    const auto& hit_type = clustering_input.inputs.at(1).int64_data;  // hit_type [N]
    const auto& h_scalar = clustering_input.inputs.at(2).float_data;  // h_scalar [N,10]
    const std::size_t n = static_cast<std::size_t>(clustering_input.batch_size);

    std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_input.txt");
    out << std::setprecision(9);
    out << n << " 14\n";  // columns: x y z hit_type h_scalar[0..9]
    for (std::size_t i = 0; i < n; ++i) {
        out << pos[3 * i + 0] << " " << pos[3 * i + 1] << " " << pos[3 * i + 2]
            << " " << hit_type[i];
        for (int c = 0; c < 10; ++c) {
            out << " " << h_scalar[10 * i + static_cast<std::size_t>(c)];
        }
        out << "\n";
    }
}

// Validation dump: raw clustering-model output (x, y, z, beta) per node.
void dumpClusteringOutput(const std::vector<float>& out0, std::size_t eventIdx) {
    const std::size_t n_rows = out0.size() / 4;
    std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_output.txt");
    out << std::setprecision(9);
    out << n_rows << " 4\n";
    for (std::size_t i = 0; i < n_rows; ++i) {
        out << out0[4 * i + 0] << " " << out0[4 * i + 1] << " " << out0[4 * i + 2]
            << " " << out0[4 * i + 3] << "\n";
    }
}

}  // namespace

 /**
  output: collection
  */






struct PFHitML final:
   k4FWCore::MultiTransformer<
      std::tuple<
      edm4hep::ReconstructedParticleCollection,
      edm4hep::ParticleIDCollection,
      RecoTruthLinkCollection
      >(

    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::CalorimeterHitCollection&,
    const edm4hep::TrackCollection&,
    const edm4hep::MCParticleCollection&, //brauche ich die noch?
    const CaloTruthLinkCollection&,
    const TrackTruthLinkCollection&

   ) > {

    PFHitML(const std::string& name, ISvcLocator* svcLoc)
      : k4FWCore::MultiTransformer<
      std::tuple<
      edm4hep::ReconstructedParticleCollection,
      edm4hep::ParticleIDCollection,
      RecoTruthLinkCollection
      > (
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::CalorimeterHitCollection&,
        const edm4hep::TrackCollection&,
        const edm4hep::MCParticleCollection&,
        const CaloTruthLinkCollection&,
        const TrackTruthLinkCollection&)>(
        name, svcLoc,
          {
            KeyValues("EcalBarrelHits", {"ECALBarrel"}),
            KeyValues("HcalBarrelHits", {"HCALBarrel"}),
            KeyValues("EcalEndcaplHits", {"ECALEndcap"}),
            KeyValues("HcalEndcapHits", {"HCALEndcap"}),
            KeyValues("HcalOtherHits", {"HCALOther"}),
            KeyValues("MUON", {"MUON"}),
            KeyValues("Tracks", {"SiTracks_Refitted"}),
            KeyValues("MCParticles", {"MCParticles"}),
            KeyValues("CalohitMCTruthLink", {"CalohitMCTruthLink"}),
            KeyValues("SiTracksMCTruthLink", {"SiTracksMCTruthLink"})
          },
          {
            KeyValues("HitPF", {"HitPF"}),
            KeyValues("HitPFIDs", {"HitPFIDs"}),  // Outputs
            KeyValues("HitPFMCTruthLink", {"HitPFMCTruthLink"})
          }  
        ) {}

  // main
  std::tuple<
  edm4hep::ReconstructedParticleCollection,
  edm4hep::ParticleIDCollection,
  RecoTruthLinkCollection
  > operator()(
    const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
    const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalEndcap_hits,
    const edm4hep::CalorimeterHitCollection& HcalOther_hits,
    const edm4hep::CalorimeterHitCollection& Muon_hits,
    const edm4hep::TrackCollection& tracks,
    const edm4hep::MCParticleCollection&,
    const CaloTruthLinkCollection& caloTruthLinks,
    const TrackTruthLinkCollection& trackTruthLinks
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

    if (m_eventCounter < m_maxDumpEvents) {
      dumpClusteringInputs(clustering_input, m_eventCounter);
    }
   
    ///////////////////////////////////////////////////
    ////////// Inference Clustering Model //////////
    ///////////////////////////////////////////////////

    info() << "================ EVENT " << m_eventCounter << " ================" << endmsg;


    
    std::vector<std::vector<float>>  outputs = m_onnx->runNamed(clustering_input.inputs);

    //get two outputs, first one has shape (N,4) (three coordinates in embedding space + beta)
    // second output is  dummy for pred_energy_corr (not needed at this stage)

    if (m_eventCounter < m_maxDumpEvents) {
      dumpClusteringOutput(outputs.at(0), m_eventCounter);
    }

    

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
    std::cout << "originallabels: " << cluster_label.sizes() << std::endl;
    std::cout << "cluster output shape: " << cluster_label_corrected.sizes() << std::endl;
    std::cout << "changed labels: " << n_changed << std::endl;

 
    //the pipeline:
    //after clustering, form graphs of hits belonging to one cluster

    //this function gets the clusters, within this function create shower instances, set x,y,z,..
    ShowerBuilder builder(extractor, inputs);
    auto showers = builder.buildShowers(cluster_label_corrected, outputs[0]);
    //std::cout << "shower output" << showers.size() <<std::endl;

    auto split = splitShowersByTrackContent(showers);

    info() << "Number of charged showers: " << split.charged.size() << endmsg;
    info() << "Number of neutral showers: " << split.neutral.size() << endmsg;

    //truth linking
    TruthMatchConfig truthCfg;
    truthCfg.iouThreshold = 0.25f;
    truthCfg.barrelRadius = 2150.f;
    truthCfg.nBarrelSides = 12;
    truthCfg.endCapZ = 2307.f;

    const auto showerTruthMatches = matchShowersByIoU(
        showers,
        caloTruthLinks,
        trackTruthLinks,
        truthCfg
    );

    

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

  auto HitPF = edm4hep::ReconstructedParticleCollection{};
  auto HitPFIDs = edm4hep::ParticleIDCollection{};
  auto HitPFMCTruthLink = RecoTruthLinkCollection{};

  const auto truthSummaries = collectTruthRecoSummaries(
    caloTruthLinks,
    trackTruthLinks,
    truthCfg
  );

  EvaluationSummaryBuilder evalBuilder(
      showers,
      showerTruthMatches,
      truthSummaries
  );



  // loop over showers per event
  for (size_t idx : split.charged) {

    //start debug
    /*
    const std::size_t edge_values = 50;

    for (const auto& input : prop_inputs[idx].inputs) {
        info() << "  name=" << input.name
              << " type=" << (input.type == ONNXInput::Type::Float ? "Float" : "Int64")
              << " shape=[";

        for (std::size_t i = 0; i < input.shape.size(); ++i) {
            if (i != 0) info() << ", ";
            info() << input.shape[i];
        }

        info() << "] data=[";

        if (input.type == ONNXInput::Type::Float) {
            const auto& data = input.float_data;
            const std::size_t n = data.size();

            if (n <= 2 * edge_values) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (i != 0) info() << ", ";
                    info() << data[i];
                }
            } else {
                for (std::size_t i = 0; i < edge_values; ++i) {
                    if (i != 0) info() << ", ";
                    info() << data[i];
                }

                info() << ", ... ";

                for (std::size_t i = n - edge_values; i < n; ++i) {
                    if (i != n - edge_values) info() << ", ";
                    info() << data[i];
                }

                info() << " (" << n << " total)";
            }
        } else {
            const auto& data = input.int64_data;
            const std::size_t n = data.size();

            if (n <= 2 * edge_values) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (i != 0) info() << ", ";
                    info() << data[i];
                }
            } else {
                for (std::size_t i = 0; i < edge_values; ++i) {
                    if (i != 0) info() << ", ";
                    info() << data[i];
                }

                info() << ", ... ";

                for (std::size_t i = n - edge_values; i < n; ++i) {
                    if (i != n - edge_values) info() << ", ";
                    info() << data[i];
                }

                info() << " (" << n << " total)";
            }
        }

        info() << "]" << endmsg;
    }
    */
    // end debug
    
  
    auto prop_outputs = m_onnx_prop_charged->runNamed(prop_inputs[idx].inputs);
    const auto& pidLogits = findPIDOutput(*m_onnx_prop_charged, prop_outputs);
    PIDPrediction pid = decodePIDLogits(pidLogits, chargedClassMap);

    ParticleRecoInfo recoInfo = buildChargedRecoInfo(showers[idx], pid.physicsClass, pid.score);

    evalBuilder.addRecoResult(idx, recoInfo);

    const auto recoIndex = HitPF.size();
    fillRecoParticle(HitPF, HitPFIDs, showers[idx], recoInfo);

    const auto reco = HitPF.at(recoIndex);
    fillRecoTruthLink(HitPFMCTruthLink, reco, showerTruthMatches[idx]);

    
    //debug

    const auto pidObj = HitPFIDs.at(recoIndex);
    const auto r = reco.getReferencePoint();
    
    info() << "=== C++ final charged properties === "
          << "idx=" << idx
          << " calibrated_E=" << reco.getEnergy()
          //<< " ref_pt=(" << r.x << ", " << r.y << ", " << r.z << ")"
          << " momentum=" << recoInfo.momentum
          << " mass=" << recoInfo.mass
          << " pid_class=" << pidObj.getType()
          << " pid_likelihood=" << pidObj.getLikelihood()
          << endmsg;

  
    
  }


  for (size_t idx : split.neutral) {


     //start debug
    /*
    const std::size_t edge_values = 50;

    for (const auto& input : prop_inputs[idx].inputs) {
        info() << "  name=" << input.name
              << " type=" << (input.type == ONNXInput::Type::Float ? "Float" : "Int64")
              << " shape=[";

        for (std::size_t i = 0; i < input.shape.size(); ++i) {
            if (i != 0) info() << ", ";
            info() << input.shape[i];
        }

        info() << "] data=[";

        if (input.type == ONNXInput::Type::Float) {
            const auto& data = input.float_data;
            const std::size_t n = data.size();

            if (n <= 2 * edge_values) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (i != 0) info() << ", ";
                    info() << data[i];
                }
            } else {
                for (std::size_t i = 0; i < edge_values; ++i) {
                    if (i != 0) info() << ", ";
                    info() << data[i];
                }

                info() << ", ... ";

                for (std::size_t i = n - edge_values; i < n; ++i) {
                    if (i != n - edge_values) info() << ", ";
                    info() << data[i];
                }

                info() << " (" << n << " total)";
            }
        } else {
            const auto& data = input.int64_data;
            const std::size_t n = data.size();

            if (n <= 2 * edge_values) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (i != 0) info() << ", ";
                    info() << data[i];
                }
            } else {
                for (std::size_t i = 0; i < edge_values; ++i) {
                    if (i != 0) info() << ", ";
                    info() << data[i];
                }

                info() << ", ... ";

                for (std::size_t i = n - edge_values; i < n; ++i) {
                    if (i != n - edge_values) info() << ", ";
                    info() << data[i];
                }

                info() << " (" << n << " total)";
            }
        }

        info() << "]" << endmsg;
    }
        */
    
    // end debug

   

    auto prop_outputs = m_onnx_prop_neutral->runNamed(prop_inputs[idx].inputs);
    const auto& pidLogits = findPIDOutput(*m_onnx_prop_neutral, prop_outputs);
    PIDPrediction pid = decodePIDLogits(pidLogits, neutralClassMap);
    float predictedEnergy = prop_outputs[0].at(0);
    edm4hep::Vector3f predictedReferencePoint = computeNeutralReferencePoint(showers[idx]);
    edm4hep::Vector3f predictedDirection = computeNeutralDirection(predictedReferencePoint);



    ParticleRecoInfo recoInfo = buildNeutralRecoInfo(showers[idx], pid.physicsClass, pid.score, predictedEnergy, predictedDirection, predictedReferencePoint);

    evalBuilder.addRecoResult(idx, recoInfo);

    const auto recoIndex = HitPF.size();
    fillRecoParticle(HitPF, HitPFIDs, showers[idx], recoInfo);

    const auto reco = HitPF.at(recoIndex);
    fillRecoTruthLink(HitPFMCTruthLink, reco, showerTruthMatches[idx]);


    //begin debug
    /*

    const auto pidObj = HitPFIDs.at(recoIndex);
    const auto r = reco.getReferencePoint();

    info() << "=== C++ final neutral properties === "
          << "idx=" << idx
          << " calibrated_E=" << reco.getEnergy()
          << " ref_pt=(" << r.x << ", " << r.y << ", " << r.z << ")"
          << " direction=" << recoInfo.direction
          << " pid_class=" << pidObj.getType()
          << " pid_likelihood=" << pidObj.getLikelihood()
          << endmsg;
    */
    //end debug




  }

  const auto evalRows = evalBuilder.finalize();


  ++m_eventCounter;
    


  return {std::move(HitPF), std::move(HitPFIDs), std::move(HitPFMCTruthLink)};  //outputs

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
  mutable std::size_t m_eventCounter{0}; //for debugging

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

  Gaudi::Property<std::size_t> m_maxDumpEvents{
    this, "maxDumpEvents", 10,
    "Number of leading events to dump validation tensors for into dump/ (0 disables dumping)"};

  


};


DECLARE_COMPONENT(PFHitML)

//cleanup? get rid of helper json thing? check if this with input name in onnx part works ous