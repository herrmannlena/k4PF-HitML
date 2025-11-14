
#include <vector>
#include <torch/torch.h>

class Clustering {
public:
    Clustering(float tbeta, float td);

    torch::Tensor get_clustering(const std::vector<float>& output_vector);


private:
    float tbeta_;
    float td_;

    torch::Tensor find_condpoints(const torch::Tensor& betas,
                                         const torch::Tensor& unassigned);

    
};
