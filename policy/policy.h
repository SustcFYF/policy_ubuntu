#ifndef LIBTORCH_POLICY_H_
#define LIBTORCH_POLICY_H_
namespace policy {
class Policy {
public:
    Policy(const std::string& model_path);
    std::vector<double> forward(std::vector<double>& state);

private:
    const int MAX_TORQUE = 10;
};
}
#endif