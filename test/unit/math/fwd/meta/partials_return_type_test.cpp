#include <stan/math/fwd.hpp>
#include <test/unit/math/util.hpp>
#include <gtest/gtest.h>
#include <vector>

using stan::partials_return_type;
using stan::math::fvar;

TEST(MathMetaFwd, PartialsReturnTypeFvarDouble) {
  stan::test::expect_same_type<double, partials_return_type<fvar<double> >::type>();
}

TEST(MathMetaFwd, PartialsReturnTypeFvarFvarDouble) {
  stan::test::expect_same_type<fvar<double>,
                         partials_return_type<fvar<fvar<double> > >::type>();
}

TEST(MathMetaFwd, PartialsReturnTypeFvarDoubleTenParams) {
  stan::test::expect_same_type<
      double,
      partials_return_type<double, fvar<double>, double, int, double, float,
                           float, float, fvar<double>, int>::type>();
}

TEST(MathMetaFwd, PartialsReturnTypeFvarFvarDoubleTenParams) {
  stan::test::expect_same_type<
      fvar<double>, partials_return_type<double, fvar<fvar<double> >, double,
                                         int, double, float, float, float,
                                         fvar<fvar<double> >, int>::type>();
}

TEST(MathMetaFwd, partials_return_type) {
  partials_return_type<double, fvar<double>, std::vector<fvar<double> > >::type
      a(5.0);
  EXPECT_EQ(5.0, a);
}
