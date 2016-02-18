#ifndef TEST_UNIT_MATH_PRIM_MAT_VECTORIZE_EXPECT_VECTOR_HPP
#define TEST_UNIT_MATH_PRIM_MAT_VECTORIZE_EXPECT_VECTOR_HPP

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

template <typename F, typename T>
void expect_vector_error() {
  using std::vector;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;
  vector<double> illegal_inputs = F::illegal_inputs();

  vector_t b = vector_t(illegal_inputs.size());
  for (size_t i = 0; i < illegal_inputs.size(); ++i) 
    b(i) = illegal_inputs[i];
  EXPECT_THROW(F::template apply<vector_t>(b), std::domain_error);

  vector<vector_t> d;
  d.push_back(b);
  d.push_back(b);
  EXPECT_THROW(F::template apply<vector<vector_t> >(d), 
               std::domain_error);
}

#endif
