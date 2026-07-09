#include "Clustering.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>

//this one for DPC https://github.com/pgoltstein/densitypeakclustering

Clustering::Clustering(float d_c, float rho_min, float delta_min, float core_radius)
    : d_c_(d_c), rho_min_(rho_min), delta_min_(delta_min), core_radius_(core_radius) {}


std::vector<float> Clustering::compute_distance_matrix(const torch::Tensor& X) const {
    const auto positions = X.accessor<float, 2>();
    const std::size_t n_points = static_cast<std::size_t>(X.size(0));
    // NaN diagonal matches Python's dc.distance_matrix (sklearn pairwise_distances
    // with the diagonal overwritten to NaN): `distance < d_c_` is false for NaN,
    // so compute_local_density() correctly excludes each hit's own energy from
    // its own density. The NaN is reset to 0 later in get_clustering(), right
    // before the core-radius membership check, exactly mirroring Python's
    // `D[np.isnan(D)] = 0` -- this makes each cluster center trivially its own
    // core member (self-distance 0) without letting self-density leak in above.
    std::vector<float> distances(n_points * n_points, std::numeric_limits<float>::quiet_NaN());
    

    for (std::size_t i = 0; i < n_points; ++i) {
        for (std::size_t j = i + 1; j < n_points; ++j) {
            const float dx = positions[i][0] - positions[j][0];
            const float dy = positions[i][1] - positions[j][1];
            const float dz = positions[i][2] - positions[j][2];
            const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            distances[i * n_points + j] = distance;
            distances[j * n_points + i] = distance;
        }
    }

    return distances;
}

std::vector<float> Clustering::compute_local_density(const std::vector<float>& distances,
                                                     const std::vector<float>& energies,
                                                     std::size_t n_points) const {
    std::vector<float> rho(n_points, 0.0f);
    for (std::size_t i = 0; i < n_points; ++i) {
        float density = 0.0f;
        for (std::size_t j = 0; j < n_points; ++j) {
            const float distance = distances[i * n_points + j];
            if (distance < d_c_) {
                const float weight = std::exp(-std::pow(distance / d_c_, 2.0f));
                density += energies[j] * weight;
            }
        }
        rho[i] = density;
    }
    return rho;
}


std::pair<std::vector<float>, std::vector<int64_t>> Clustering::distance_to_larger_density(
    const std::vector<float>& distances,
    const std::vector<float>& rho,
    std::size_t n_points) const {
    std::vector<float> delta(n_points, 0.0f);
    std::vector<int64_t> nearest(n_points, -1);

    if (n_points == 0) {
        return {delta, nearest};
    }

    std::vector<std::size_t> order(n_points);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&rho](std::size_t lhs, std::size_t rhs) {
        return rho[lhs] > rho[rhs];
    });

    float max_distance = 0.0f;
    for (float distance : distances) {
        max_distance = std::max(max_distance, distance);
    }

    for (std::size_t rank = 0; rank < n_points; ++rank) {
        const std::size_t index = order[rank];
        float best_distance = std::numeric_limits<float>::max();
        int64_t best_neighbor = -1;

        // The sort bounds the search to a safe superset of "strictly higher
        // rho" candidates (ranks before this one); the explicit inequality
        // below is what actually decides membership, so ties in rho (e.g.
        // several isolated zero-energy hits) are never treated as
        // higher-density in either direction -- matching Python's strict
        // `rho[j] > rho[i]` test exactly, independent of how std::sort
        // happens to order the tied entries.
        for (std::size_t higher_rank = 0; higher_rank < rank; ++higher_rank) {
            const std::size_t higher_index = order[higher_rank];
            if (!(rho[higher_index] > rho[index])) {
                continue;
            }
            const float distance = distances[index * n_points + higher_index];
            if (distance < best_distance) {
                best_distance = distance;
                best_neighbor = static_cast<int64_t>(higher_index);
            }
        }

        if (best_neighbor == -1) {
            // No point with strictly higher density (the global peak, or
            // anything tied with it) falls back to the max pairwise distance
            // in the event and is left unassigned (-1) in `nearest`. Such
            // points always fail the rho > rho_min gate in cluster_centers()
            // whenever more than one point ties for the top density, so this
            // fallback never actually changes which points become centers.
            delta[index] = max_distance;
        } else {
            delta[index] = best_distance;
            nearest[index] = best_neighbor;
        }
    }

    return {delta, nearest};
}

std::vector<int64_t> Clustering::cluster_centers(const std::vector<float>& rho,
                                                 const std::vector<float>& delta) const {
    std::vector<int64_t> centers;
    for (std::size_t i = 0; i < rho.size(); ++i) {
        if (rho[i] > rho_min_ && delta[i] > delta_min_) {
            centers.push_back(static_cast<int64_t>(i));
        }
    }
    std::stable_sort(centers.begin(), centers.end(), [&rho](int64_t lhs, int64_t rhs) {
        return rho[static_cast<std::size_t>(lhs)] > rho[static_cast<std::size_t>(rhs)];
    });
    return centers;
}

std::vector<int64_t> Clustering::assign_cluster_id(const std::vector<float>& rho,
                                                   const std::vector<int64_t>& nearest,
                                                   const std::vector<int64_t>& centers) const {
    std::vector<int64_t> ids(rho.size(), -1);
    for (std::size_t i = 0; i < centers.size(); ++i) {
        ids[centers[i]] = static_cast<int64_t>(i);
    }

    std::vector<std::size_t> order(rho.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&rho](std::size_t lhs, std::size_t rhs) {
        return rho[lhs] > rho[rhs];
    });

    for (std::size_t index : order) {
        if (ids[index] != -1) {
            continue;
        }
        const int64_t parent = nearest[index];
        if (parent >= 0) {
            ids[index] = ids[static_cast<std::size_t>(parent)];
        }
    }

    return ids;
}

torch::Tensor Clustering::get_clustering(const std::vector<float>& output_vector,
                                         const std::vector<float>& energies,
                                         int64_t dumpEventIdx) {
    // Expect 4 values per row: [x, y, z, beta]
    if (output_vector.size() % 4 != 0)
        throw std::runtime_error("Flat model output size is not divisible by 4");

    const int64_t N = output_vector.size() / 4;
    if (energies.size() != static_cast<std::size_t>(N)) {
        throw std::runtime_error("Energy vector size does not match model output rows");
    }

    // Convert to (N,4) torch tensor
    torch::Tensor output_model_tensor = torch::from_blob(
        const_cast<float*>(output_vector.data()),
        {N, 4},
        torch::dtype(torch::kFloat32)
    ).clone();

    torch::Tensor X = output_model_tensor.slice(1, 0, 3).contiguous();
    const std::size_t n_points = static_cast<std::size_t>(N);


    auto distances = compute_distance_matrix(X);
    const auto rho = compute_local_density(distances, energies, n_points);
    const auto [delta, nearest] = distance_to_larger_density(distances, rho, n_points);
    const auto centers = cluster_centers(rho, delta);
    const auto ids = assign_cluster_id(rho, nearest, centers);

    
    // Mimic Python behavior: D has NaNs during rho calculation, then
    // D[np.isnan(D)] = 0 before core assignment.

    for (float& d : distances) {
        if (std::isnan(d)) {
            d = 0.0f;
        }
    }


    std::vector<int64_t> labels(n_points, 0);
    for (std::size_t center_index = 0; center_index < centers.size(); ++center_index) {
        const int64_t center = centers[center_index];
        for (std::size_t point = 0; point < n_points; ++point) {
            if (ids[point] == static_cast<int64_t>(center_index) &&
                distances[point * n_points + static_cast<std::size_t>(center)] < core_radius_) {
                labels[point] = static_cast<int64_t>(center_index) + 1;
            }
        }
    }

    // Validation dump: per-hit DPC intermediates (rho, delta, nearest, id,
    // label) plus the list of cluster-center hit indices, in ID order.
    if (dumpEventIdx >= 0) {
        std::ofstream out("dump/cpp_event_" + std::to_string(dumpEventIdx) + "_dpc.txt");
        out << std::setprecision(9);
        out << n_points << " 5\n";  // columns: rho delta nearest id label
        for (std::size_t i = 0; i < n_points; ++i) {
            out << rho[i] << " " << delta[i] << " " << nearest[i] << " "
                << ids[i] << " " << labels[i] << "\n";
        }

        std::ofstream centersOut(
            "dump/cpp_event_" + std::to_string(dumpEventIdx) + "_dpc_centers.txt");
        centersOut << centers.size() << " 1\n";
        for (int64_t c : centers) {
            centersOut << c << "\n";
        }
    }

    return torch::tensor(labels, torch::dtype(torch::kLong));
}


//for clustering post-processing
torch::Tensor Clustering::remove_bad_tracks_from_cluster(
    const torch::Tensor& labels_in,
    const std::vector<float>& hit_type,
    const std::vector<float>& e_hits,
    const std::vector<float>& p_hits
) {
    auto labels = labels_in.clone().to(torch::kLong);

    const int64_t n_nodes = labels.size(0);


    if (hit_type.size() != static_cast<std::size_t>(n_nodes) ||
        e_hits.size()    != static_cast<std::size_t>(n_nodes) ||
        p_hits.size()    != static_cast<std::size_t>(n_nodes)) {
        throw std::runtime_error("remove_bad_tracks_from_cluster: input sizes do not match labels");
    }

    const int64_t max_label = labels.max().item<int64_t>();
    auto labels_acc = labels.accessor<int64_t, 1>();

    for (int64_t cluster_id = 1; cluster_id <= max_label; ++cluster_id) {
        float e_cluster = 0.0f;
        int n_muon_hits = 0;
        std::vector<int64_t> track_nodes;


        for (int64_t node = 0; node < n_nodes; ++node) {
            if (labels_acc[node] != cluster_id) continue;

            e_cluster += e_hits[node];

            if (hit_type[node] == 4) {
                n_muon_hits++;
            }
            if (hit_type[node] == 1) {
                track_nodes.push_back(node);
            }
        }

        if (track_nodes.empty()) continue;

        for (auto node : track_nodes) {
            const float p_track = p_hits[node];
            if (p_track <= 0.0f) continue;

            const float diff = std::abs(e_cluster - p_track) / p_track;
            const float sigma_4 = 4.0f * 0.5f / std::sqrt(p_track);
            const bool bad_track = (diff > sigma_4) && (n_muon_hits < 1);

            if (bad_track) {
                labels_acc[node] = 0;
            }
        }
    }


    return labels;
}
