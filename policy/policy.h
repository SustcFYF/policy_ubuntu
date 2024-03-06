#include "torch/torch.h"
#include "torch/script.h"

class Policy {
public:
    Policy(const std::string& model_path);
    at::Tensor forward(torch::Tensor& state);

private:
    torch::jit::script::Module model;
    const int MAX_TORQUE = 10;
};