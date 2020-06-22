#ifndef STAN_MATH_REV_CORE_PROFILING_HPP
#define STAN_MATH_REV_CORE_PROFILING_HPP

#include <stan/math/rev/core/chainablestack.hpp>
#include <chrono>
#include <string>
#include <unordered_map>

namespace stan {
namespace math {
namespace profiling {    

typedef struct {
  std::chrono::time_point<std::chrono::steady_clock> forward_pass_time_start_;
  std::chrono::time_point<std::chrono::steady_clock> backward_pass_time_start_;
  bool forward_pass_running_;
  bool backward_pass_running_;
  size_t start_var_stack_size_;
  size_t stop_var_stack_size_;
  size_t var_stack_used_;
  double forward_pass_time_;
  double backward_pass_time_;
  size_t forward_pass_iterations_;
  size_t backward_pass_iterations_;
} profiler_state;

using profiler_map = std::unordered_map<std::string, profiler_state>;

inline void start_forward_pass(std::string id, profiler_map& p) {
    if (p.count(id) == 0) {
        p[id].forward_pass_iterations_ = 0;
        p[id].backward_pass_iterations_ = 0;
        p[id].forward_pass_time_ = 0;
        p[id].backward_pass_time_ = 0;
    }
    p[id].forward_pass_iterations_++;
    p[id].forward_pass_running_ = true;
    p[id].forward_pass_time_start_ = std::chrono::steady_clock::now();
    p[id].start_var_stack_size_ = ChainableStack::instance_->var_stack_.size();
}

inline void stop_forward_pass(std::string id, profiler_map& p) {
    p[id].forward_pass_running_ = false;
    p[id].forward_pass_time_ += (std::chrono::steady_clock::now() - p[id].forward_pass_time_start_).count();
    p[id].stop_var_stack_size_ = ChainableStack::instance_->var_stack_.size();
}

}  // namespace profiling
}  // namespace math
}  // namespace stan

#endif
