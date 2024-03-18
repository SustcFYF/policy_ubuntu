#include <iostream>
//#include <aris.hpp>
#include "policy/policy.h"

int main() {
    // 加载模型
    // double test[6]{0};
    // aris::dynamic::dsp(1, 6, test);
    // aris::server::ControlServer& cs = aris::server::ControlServer::instance();
    // aris::core::fromXmlFile(cs, "test");
    std::string model_path = "../model/022714_real_tip_[0,0.725]_traced.pt";
    policy::Policy policy(model_path);

    // 输入数据: [j0p, j1p, j2p, j0v, j1v, j2v, x, y, q, vx, vy, wz, j1f, j2f]
    std::vector<double> state = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    std::cout << "state: " << state << std::endl;

    // 前向传播
    std::vector<double> torque = policy.forward(state);

    // 输出结果
    std::cout << "torque: " << torque << std::endl;
}


