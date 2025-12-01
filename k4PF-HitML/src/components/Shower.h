
#ifndef SHOWER_H
#define SHOWER_H

//edm4hep imports
#include "edm4hep/TrackCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"

// shower object, assig energy, direction, PID..

class Shower {
public:
    void addCalorimeterHit(const edm4hep::CalorimeterHit& hit, const std::string collection = "");

    void addTrack(const edm4hep::Track& track) {
        tracks_.push_back(track);
        types_.push_back(1);
    }

    void addBetas(float beta) {
        betas_.push_back(beta);
    }

    const std::vector<edm4hep::CalorimeterHit>& getCalorimeterHits() const {
        return caloHits_;
    }

    const std::vector<edm4hep::Track>& getTracks() const {
        return tracks_;
    }

    const std::pair<float, float> getCaloEnergy(std::vector<edm4hep::CalorimeterHit> collection);

    const float getTrackMomentum_mean(); //retruns mean track momentum per shower

    const float Chi2_mean();

    const std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> get_pos();

    const std::tuple<std::vector<float>, std::vector<float>> get_ep();

    // later: add tracks, energy sums, etc.

    std::vector<edm4hep::CalorimeterHit> caloHits_;
    std::vector<edm4hep::CalorimeterHit> ecalHits_;
    std::vector<edm4hep::CalorimeterHit> hcalHits_;
    std::vector<edm4hep::CalorimeterHit> muonHits_;
    std::vector<edm4hep::Track> tracks_;

    std::vector<int> types_;
    std::vector<float> betas_;


    
};


#endif // SHOWER_H
