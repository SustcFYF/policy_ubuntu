#include "policy.h"

#include "torch/torch.h"
#include "torch/script.h"

namespace policy {

static torch::jit::script::Module model;

Policy::Policy(const std::string& model_path){
    model = torch::jit::load(model_path, torch::kCPU);
}

auto Policy::forward(std::vector<double>& state)->std::vector<double> {
    // 将输入状态转换为 Torch 张量
    torch::Tensor state_tensor = torch::tensor(state).reshape({1, static_cast<long int>(state.size())});
    // 进行前向推理
    torch::Tensor output_tensor = model.forward({state_tensor}).toTensor() * max_torque_;
    // 将 Torch 张量转换为 std::vector<double>
    std::vector<double> output(output_tensor.data_ptr<float>(), output_tensor.data_ptr<float>() + output_tensor.numel());

    // 返回结果
    return output;
}

auto Policy::setMaxTorque(double value)->void{
    max_torque_ = value;
}

auto Policy::maxTorque()->double{
    return max_torque_;
}
}