#ifndef LIBTORCH_POLICY_H_
#define LIBTORCH_POLICY_H_
#include <string>
#include <vector>

namespace policy {
class Policy {
public:
    Policy(const std::string& model_path);
    auto forward(std::vector<double>& state)->std::vector<double>;
    auto setMaxTorque(double value=10.0)->void;
    auto maxTorque()->double;

private:
    double max_torque_{10.0};
};
}
#endif