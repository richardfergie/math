#include <test/unit/math/test_ad.hpp>
#include <vector>

auto f(int i) {
  return [=](const auto& y) { return stan::math::head(y, i).eval(); };
}

auto g(int i) {
  return [=](const auto& y) { return stan::math::head(y, i); };
}

template <typename T>
void expect_head(const T& x, int n) {
  std::vector<std::vector<double>> stx{x, x, x};
  Eigen::VectorXd v = stan::test::to_vector(x);
  std::vector<Eigen::VectorXd> stv{v, v, v};
  Eigen::RowVectorXd rv = stan::test::to_row_vector(x);
  std::vector<Eigen::RowVectorXd> strv{rv, rv, rv};
  stan::test::expect_ad(g(n), x);
  stan::test::expect_ad(g(n), stx);
  stan::test::expect_ad(f(n), v);
  stan::test::expect_ad(g(n), stv);
  stan::test::expect_ad(f(n), rv);
  stan::test::expect_ad(g(n), strv);
}

TEST(MathMixMatFun, head) {
  std::vector<double> a{};
  expect_head(a, 0);
  expect_head(a, 1);

  std::vector<double> b{1};
  expect_head(b, 0);
  expect_head(b, 1);
  expect_head(b, 2);

  std::vector<double> v{1, 2, 3};
  for (int n = 0; n < 5; ++n) {
    expect_head(v, n);
  }
}
