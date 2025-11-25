
#ifndef SHOWER_H
#define SHOWER_H

// shower object, assig energy, direction, PID..

class Shower {
public:
    void addCalorimeterHit(const edm4hep::CalorimeterHit& hit) {
        caloHits_.push_back(&hit);
    }

    void addTrack(const edm4hep::Track& track) {
        tracks_.push_back(&track);
    }

    const std::vector<const edm4hep::CalorimeterHit*>& getCalorimeterHits() const {
        return caloHits_;
    }

    // later: add tracks, energy sums, etc.

private:
    std::vector<const edm4hep::CalorimeterHit*> caloHits_;
    std::vector<const edm4hep::Track*> tracks_;
};


#endif // SHOWER_H
