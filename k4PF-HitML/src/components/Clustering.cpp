#include "Clustering.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

//this one for DPC https://github.com/pgoltstein/densitypeakclustering

Clustering::Clustering(float d_c, float rho_min, float delta_min, float core_radius)
    : d_c_(d_c), rho_min_(rho_min), delta_min_(delta_min), core_radius_(core_radius) {}


std::vector<float> Clustering::compute_distance_matrix(const torch::Tensor& X) const {
    const auto positions = X.accessor<float, 2>();
    const std::size_t n_points = static_cast<std::size_t>(X.size(0));
    std::vector<float> distances(n_points * n_points, 0.0f);

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

//this function could differ -> check
std::pair<std::vector<float>, std::vector<int64_t>> Clustering::distance_to_larger_density(
    const std::vector<float>& distances,
    const std::vector<float>& rho,
    std::size_t n_points) const {
    std::vector<float> delta(n_points, 0.0f);
    std::vector<int64_t> nearest(n_points, -1);
    std::vector<std::size_t> order(n_points);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&rho](std::size_t lhs, std::size_t rhs) {
        return rho[lhs] > rho[rhs];
    });

    float max_distance = 0.0f;
    for (float distance : distances) {
        max_distance = std::max(max_distance, distance);
    }

    if (n_points == 0) {
        return {delta, nearest};
    }

    delta[order[0]] = max_distance;
    nearest[order[0]] = static_cast<int64_t>(order[0]);

    for (std::size_t rank = 1; rank < n_points; ++rank) {
        const std::size_t index = order[rank];
        float best_distance = std::numeric_limits<float>::max();
        int64_t best_neighbor = -1;

        for (std::size_t higher_rank = 0; higher_rank < rank; ++higher_rank) {
            const std::size_t higher_index = order[higher_rank];
            const float distance = distances[index * n_points + higher_index];
            if (distance < best_distance) {
                best_distance = distance;
                best_neighbor = static_cast<int64_t>(higher_index);
            }
        }

        delta[index] = best_distance;
        nearest[index] = best_neighbor;
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
                                         const std::vector<float>& energies) {
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

    const auto distances = compute_distance_matrix(X);
    const auto rho = compute_local_density(distances, energies, n_points);
    const auto [delta, nearest] = distance_to_larger_density(distances, rho, n_points);
    const auto centers = cluster_centers(rho, delta);
    const auto ids = assign_cluster_id(rho, nearest, centers);

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

    return torch::tensor(labels, torch::dtype(torch::kLong));
}
