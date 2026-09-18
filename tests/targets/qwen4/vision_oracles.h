#pragma once
#include <vector>
namespace ninfer::test::vision_source {
// Test-only adapters to the existing mathematical Op oracles, accepting double ideals.
std::vector<double> norm(const std::vector<double>&,const std::vector<float>&,
                         const std::vector<float>&,int);
std::vector<double> rotary(const std::vector<double>&,const std::vector<int>&,int);
std::vector<double> attention(const std::vector<double>&,const std::vector<double>&,
                              const std::vector<double>&,const std::vector<int>&);
}
