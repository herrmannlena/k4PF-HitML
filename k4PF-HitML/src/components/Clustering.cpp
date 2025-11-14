#include "Clustering.h"


Clustering::Clustering(float tbeta, float td)
    : tbeta_(tbeta), td_(td) {}



torch::Tensor Clustering::find_condpoints(const torch::Tensor& betas,
                                               const torch::Tensor& unassigned) 
{
    torch::Tensor mask_unassigned =
        torch::zeros_like(betas, torch::dtype(torch::kBool));
    mask_unassigned.index_put_({unassigned}, true);

    torch::Tensor select_condpoints = (betas > tbeta_) & mask_unassigned;
    torch::Tensor indices_condpoints =
        torch::nonzero(select_condpoints).view({-1});

    if (indices_condpoints.numel() == 0) {
        return torch::empty({0}, torch::dtype(torch::kLong));
    }

    torch::Tensor betas_condpoints = betas.index_select(0, indices_condpoints);
    torch::Tensor sorted_indices =
        std::get<1>(betas_condpoints.sort(0, true));

    return indices_condpoints.index_select(0, sorted_indices);
}


torch::Tensor Clustering::get_clustering(const std::vector<float>& output_vector) 
{
    // Expect 4 values per row: [x, y, z, beta]
    if (output_vector.size() % 4 != 0)
        throw std::runtime_error("Flat model output size is not divisible by 4");

    const int64_t N = output_vector.size() / 4;

    // Convert to (N,4) torch tensor
    torch::Tensor output_model_tensor = torch::from_blob(
        const_cast<float*>(output_vector.data()),
        {N, 4},
        torch::dtype(torch::kFloat32)
    ).clone();

    // Extract X (xyz) and betas
    torch::Tensor X     = output_model_tensor.slice(1, 0, 3);
    torch::Tensor betas = output_model_tensor.select(1, 3);

    int64_t n_points = betas.size(0);
    torch::Tensor clustering =
        torch::zeros({n_points}, torch::dtype(torch::kLong));
    torch::Tensor unassigned =
        torch::arange(n_points, torch::dtype(torch::kLong));

    int64_t index_assignation = 1;
    torch::Tensor indices_condpoints =
        find_condpoints(betas, unassigned);

    while (indices_condpoints.numel() > 0 && unassigned.numel() > 0) {
        int64_t index_condpoint = indices_condpoints[0].item<int64_t>();
        torch::Tensor coord_cond = X[index_condpoint];

        torch::Tensor dists =
            torch::norm(X.index_select(0, unassigned) - coord_cond, 2, 1);
        torch::Tensor mask_distance = dists < td_;

        if (mask_distance.sum().item<int64_t>() == 0) {
            // Remove first condpoint and try next
            indices_condpoints = indices_condpoints.slice(0, 1);
            continue;
        }

        // Assign cluster
        torch::Tensor assigned_points =
            torch::masked_select(unassigned, mask_distance);
        clustering.index_put_({assigned_points}, index_assignation);

        // Remove assigned points
        torch::Tensor mask_keep = ~mask_distance;
        unassigned = torch::masked_select(unassigned, mask_keep);

        // Recompute conditional points
        indices_condpoints = find_condpoints(betas, unassigned);
        index_assignation += 1;
    }

    return clustering;
}