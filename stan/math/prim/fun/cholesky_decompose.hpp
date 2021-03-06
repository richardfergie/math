#ifndef STAN_MATH_PRIM_FUN_CHOLESKY_DECOMPOSE_HPP
#define STAN_MATH_PRIM_FUN_CHOLESKY_DECOMPOSE_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#ifdef STAN_OPENCL
#include <stan/math/opencl/opencl.hpp>
#endif

#include <cmath>

namespace stan {
namespace math {

/**
 * Return the lower-triangular Cholesky factor (i.e., matrix
 * square root) of the specified square, symmetric matrix.  The return
 * value \f$L\f$ will be a lower-triangular matrix such that the
 * original matrix \f$A\f$ is given by
 * <p>\f$A = L \times L^EigMat\f$.
 *
 * @tparam EigMat type of the matrix (must be derived from \c Eigen::MatrixBase)
 * @param m Symmetric matrix.
 * @return Square root of matrix.
 * @note Because OpenCL only works on doubles there are two
 * <code>cholesky_decompose</code> functions. One that works on doubles
 * and another that works on all other types (this one).
 * @throw std::domain_error if m is not a symmetric matrix or
 *   if m is not positive definite (if m has more than 0 elements)
 */
template <typename EigMat, require_eigen_t<EigMat>* = nullptr,
          require_not_vt_same<double, EigMat>* = nullptr,
          require_not_eigen_vt<is_var, EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, EigMat::RowsAtCompileTime,
                     EigMat::ColsAtCompileTime>
cholesky_decompose(const EigMat& m) {
  const eval_return_type_t<EigMat>& m_eval = m.eval();
  check_symmetric("cholesky_decompose", "m", m_eval);
  check_not_nan("cholesky_decompose", "m", m_eval);
  Eigen::LLT<Eigen::Matrix<value_type_t<EigMat>, EigMat::RowsAtCompileTime,
                           EigMat::ColsAtCompileTime>>
      llt = m_eval.llt();
  check_pos_definite("cholesky_decompose", "m", llt);
  return llt.matrixL();
}

/**
 * Return the lower-triangular Cholesky factor (i.e., matrix
 * square root) of the specified square, symmetric matrix.  The return
 * value \f$L\f$ will be a lower-triangular matrix such that the
 * original matrix \f$A\f$ is given by
 * <p>\f$A = L \times L^EigMat\f$.
 *
 * @tparam EigMat type of the matrix (must be derived from \c Eigen::MatrixBase)
 * @param m Symmetric matrix.
 * @return Square root of matrix.
 * @note Because OpenCL only works on doubles there are two
 * <code>cholesky_decompose</code> functions. One that works on doubles
 * (this one) and another that works on all other types.
 * @throw std::domain_error if m is not a symmetric matrix or
 *   if m is not positive definite (if m has more than 0 elements)
 */
template <typename EigMat, require_eigen_t<EigMat>* = nullptr,
          require_vt_same<double, EigMat>* = nullptr>
inline Eigen::Matrix<double, EigMat::RowsAtCompileTime,
                     EigMat::ColsAtCompileTime>
cholesky_decompose(const EigMat& m) {
  const eval_return_type_t<EigMat>& m_eval = m.eval();
  check_not_nan("cholesky_decompose", "m", m_eval);
#ifdef STAN_OPENCL
  if (m.rows() >= opencl_context.tuning_opts().cholesky_size_worth_transfer) {
    matrix_cl<double> m_cl(m_eval);
    return from_matrix_cl(cholesky_decompose(m_cl));
  } else {
    check_symmetric("cholesky_decompose", "m", m_eval);
    Eigen::LLT<Eigen::Matrix<double, EigMat::RowsAtCompileTime,
                             EigMat::ColsAtCompileTime>>
        llt = m_eval.llt();
    check_pos_definite("cholesky_decompose", "m", llt);
    return llt.matrixL();
  }
#else
  check_symmetric("cholesky_decompose", "m", m_eval);
  Eigen::LLT<Eigen::Matrix<double, EigMat::RowsAtCompileTime,
                           EigMat::ColsAtCompileTime>>
      llt = m_eval.llt();
  check_pos_definite("cholesky_decompose", "m", llt);
  return llt.matrixL();
#endif
}

}  // namespace math
}  // namespace stan

#endif
