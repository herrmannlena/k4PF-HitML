
#include <vector>
#include <cstddef>
#include <torch/torch.h>

class Clustering {
public:
    Clustering(float d_c, float rho_min, float delta_min, float core_radius = 0.5f);

    torch::Tensor get_clustering(const std::vector<float>& output_vector,
                                 const std::vector<float>& energies);


private:
    float d_c_;
    float rho_min_;
    float delta_min_;
    float core_radius_;

    std::vector<float> compute_distance_matrix(const torch::Tensor& X) const;
    std::vector<float> compute_local_density(const std::vector<float>& distances,
                                             const std::vector<float>& energies,
                                             std::size_t n_points) const;
    std::pair<std::vector<float>, std::vector<int64_t>> distance_to_larger_density(
        const std::vector<float>& distances,
        const std::vector<float>& rho,
        std::size_t n_points) const;
    std::vector<int64_t> cluster_centers(const std::vector<float>& rho,
                                         const std::vector<float>& delta) const;
    std::vector<int64_t> assign_cluster_id(const std::vector<float>& rho,
                                           const std::vector<int64_t>& nearest,
                                           const std::vector<int64_t>& centers) const;
};
