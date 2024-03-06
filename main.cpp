#include "policy/policy.h"
#include <iostream>

int main() {
    // 加载模型
    std::string model_path = "../model/022714_real_tip_[0,0.725]_traced.pt";
    Policy policy(model_path);

    // 输入数据: [j0p, j1p, j2p, j0v, j1v, j2v, x, y, q, vx, vy, wz, j1f, j2f]
    torch::Tensor state = torch::rand({ 1, 14 });
    std::cout << "state: " << state << std::endl;

    // 前向传播
    torch::Tensor torque = policy.forward(state);

    // 输出结果
    std::cout << "torque: " << torque << std::endl;
}


