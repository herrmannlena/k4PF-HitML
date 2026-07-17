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

#include <cstddef>
#include <torch/torch.h>
#include <vector>

class Clustering {
public:
  Clustering(float d_c, float rho_min, float delta_min, float core_radius = 0.5f);

  torch::Tensor get_clustering(const std::vector<float>& output_vector, const std::vector<float>& energies,
                               int64_t dumpEventIdx = -1);

  torch::Tensor remove_bad_tracks_from_cluster(const torch::Tensor& labels_in, const std::vector<float>& hit_type,
                                               const std::vector<float>& e_hits, const std::vector<float>& p_hits);

private:
  float d_c_;
  float rho_min_;
  float delta_min_;
  float core_radius_;

  std::vector<float> compute_distance_matrix(const torch::Tensor& X) const;
  std::vector<float> compute_local_density(const std::vector<float>& distances, const std::vector<float>& energies,
                                           std::size_t n_points) const;
  std::pair<std::vector<float>, std::vector<int64_t>> distance_to_larger_density(const std::vector<float>& distances,
                                                                                 const std::vector<float>& rho,
                                                                                 std::size_t n_points) const;
  std::vector<int64_t> cluster_centers(const std::vector<float>& rho, const std::vector<float>& delta) const;
  std::vector<int64_t> assign_cluster_id(const std::vector<float>& rho, const std::vector<int64_t>& nearest,
                                         const std::vector<int64_t>& centers) const;
};
