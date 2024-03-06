#include "policy.h"

Policy::Policy(const std::string& model_path){
    model = torch::jit::load(model_path, torch::kCPU);
}

torch::Tensor Policy::forward(torch::Tensor& state) {
    return model.forward({ state }).toTensor() * MAX_TORQUE;
}