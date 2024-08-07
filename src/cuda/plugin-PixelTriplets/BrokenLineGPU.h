#ifndef RecoPixelVertexing_PixelTrackFitting_interface_BrokenLineGPU_h
#define RecoPixelVertexing_PixelTrackFitting_interface_BrokenLineGPU_h

#include <Eigen/Eigenvalues>

#include "FitUtils.h"
//#include "MatMulABA.h"
#include "defs.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace BrokenLine4 {

  namespace cg = cooperative_groups;

  //!< Karimäki's parameters: (phi, d, k=1/R)
  /*!< covariance matrix: \n
    |cov(phi,phi)|cov( d ,phi)|cov( k ,phi)| \n
    |cov(phi, d )|cov( d , d )|cov( k , d )| \n
    |cov(phi, k )|cov( d , k )|cov( k , k )|
  */
  using karimaki_circle_fit = Rfit::circle_fit;

  /*!
    \brief data needed for the Broken Line fit procedure.
  */
  template <int N>
  struct PreparedBrokenLineData {
    int q;                      //!< particle charge
    Rfit::Matrix2xNd<N> radii;  //!< xy data in the system in which the pre-fitted center is the origin
    Rfit::VectorNd<N> s;        //!< total distance traveled in the transverse plane
                                //   starting from the pre-fitted closest approach
    Rfit::VectorNd<N> S;        //!< total distance traveled (three-dimensional)
    Rfit::VectorNd<N> Z;        //!< orthogonal coordinate to the pre-fitted line in the sz plane
    Rfit::VectorNd<N> VarBeta;  //!< kink angles in the SZ plane
  };

  /*!
    \brief Computes the Coulomb multiple scattering variance of the planar angle.

    \param length length of the track in the material.
    \param B magnetic field in Gev/cm/c.
    \param R radius of curvature (needed to evaluate p).
    \param Layer denotes which of the four layers of the detector is the endpoint of the multiple scattered track. For example, if Layer=3, then the particle has just gone through the material between the second and the third layer.

    \todo add another Layer variable to identify also the start point of the track, so if there are missing hits or multiple hits, the part of the detector that the particle has traversed can be exactly identified.

    \warning the formula used here assumes beta=1, and so neglects the dependence of theta_0 on the mass of the particle at fixed momentum.

    \return the variance of the planar angle ((theta_0)^2 /3).
  */
  __device__ inline double MultScatt(const double& length,
                                     const double B,
                                     const double R,
                                     int Layer,
                                     double slope) {
    // limit R to 20GeV...
    auto pt2 = std::min(20., B * R);
    pt2 *= pt2;
    constexpr double XXI_0 = 0.06 / 16.;  //!< inverse of radiation length of the material in cm

    constexpr double geometry_factor =
        0.7;  //!< number between 1/3 (uniform material) and 1 (thin scatterer) to be manually tuned
    constexpr double fact = geometry_factor * Rfit::sqr(13.6 / 1000.);
    return fact / (pt2 * (1. + Rfit::sqr(slope))) * (std::abs(length) * XXI_0) *
           Rfit::sqr(1. + 0.038 * log(std::abs(length) * XXI_0));
  }

  /*!
    \brief Computes the 2D rotation matrix that transforms the line y=slope*x into the line y=0.

    \param slope tangent of the angle of rotation.

    \return 2D rotation matrix.
  */
  __device__ inline Rfit::Matrix2d RotationMatrix(double slope) {
    Rfit::Matrix2d Rot;
    Rot(0, 0) = 1. / sqrt(1. + Rfit::sqr(slope));
    Rot(0, 1) = slope * Rot(0, 0);
    Rot(1, 0) = -Rot(0, 1);
    Rot(1, 1) = Rot(0, 0);
    return Rot;
  }

  template <typename MATRIX, typename TILE, unsigned const int N>
  __device__ inline void ABAteqC(MATRIX const& A,
                                 MATRIX const& B,
                                 MATRIX& C,
                                 MATRIX& holder,
                                 TILE& tile) {

    auto idx = tile.thread_rank();
    double tmp;
    unsigned int row;
    unsigned int col;

    for (auto i = idx; i < N * N; i += tile.num_threads()) {
      row = i / N;
      col = i % N;
      tmp = 0.f;
#pragma unroll
      for (unsigned int m = 0; m < N; m++) {
#pragma unroll
        for (unsigned int k = 0; k < N; k++) {
          tmp += A(row, k) * B(k, m) * A(col, m);
        }
      }
      holder(row, col) = tmp;
    }

    C = holder;
   tile.sync();
  }

  template <typename MATRIX,
            typename TILE,
            typename M,
            unsigned const int N>
  __device__ inline void ABeqC(M const& A,
                               MATRIX const& B,
                               M& C,
                               TILE& tile) {
    auto idx = tile.thread_rank();

    double tmp;
    unsigned int row;
    unsigned int col;

    for (auto i = idx; i < N * N; i += tile.num_threads()) {
      row = i / N;
      col = i % N;
      tmp = 0;

#pragma unroll
      for (unsigned int k = 0; k < N; k++) {
        tmp += A(row, k) * B(k, col);
      }

      C(row, col) = tmp;
    }
    //tile.sync();
  }

  template <typename MATRIX,
            typename TILE,
            typename M,
            unsigned const int N>
  __device__ inline void ABteqC(M const& A,
                                M const& B,
                                MATRIX& C,
                                TILE& tile) {
    auto idx = tile.thread_rank();

    double tmp;
    unsigned int row;
    unsigned int col;

    for (auto i = idx; i < N * N; i += tile.num_threads()) {
      row = i / N;
      col = i % N;
      tmp = 0;

#pragma unroll
      for (unsigned int k = 0; k < N; k++) {
        tmp += A(row, k) * B(col, k);
      }

      C(row, col) = tmp;
    }
   // tile.sync();
  }

  template <typename MATRIX,
            typename M,
            typename TILE>
  __device__ inline void jacobiMult(M const& A,
                                    MATRIX const& B,
                                    MATRIX& C,
                                    M& holder,
                                    TILE& tile) {
#ifdef __MULTIPLY_MULTIPLE_STEPS_PARALLEL
    ABeqC<MATRIX, TILE, M, MATRIX::ColsAtCompileTime>(A, B, holder, tile);
    ABteqC<MATRIX, TILE, M, MATRIX::ColsAtCompileTime>(holder, A, C, tile);
#endif
#ifdef __MULTIPLY_ONE_STEP_PARALLEL
    ABAteqC<MATRIX, TILE, MATRIX::ColsAtCompileTime>(A, B, C, holder, tile);
#endif
#ifdef __MULTIPLY_SERIAL
    C = A * B * A.transpose();
#endif
  }

  /*!
    \brief Changes the Karimäki parameters (and consequently their covariance matrix) under a translation of the coordinate system, such that the old origin has coordinates (x0,y0) in the new coordinate system. The formulas are taken from Karimäki V., 1990, Effective circle fitting for particle trajectories, Nucl. Instr. and Meth. A305 (1991) 187.

    \param circle circle fit in the old coordinate system.
    \param x0 x coordinate of the translation vector.
    \param y0 y coordinate of the translation vector.
    \param jacobian passed by reference in order to save stack.
  */
  template <unsigned int TileSize,
            typename M3x3>
  __device__ inline void TranslateKarimaki(karimaki_circle_fit& circle,
                                           double x0,
                                           double y0,
                                           M3x3& jacobian,
                                           M3x3& holder,
                                           cg::thread_block_tile<TileSize>& tile) {
    double A, U, BB, C, DO, DP, uu, xi, v, mu, lambda, zeta;
    DP = x0 * cos(circle.par(0)) + y0 * sin(circle.par(0));
    DO = x0 * sin(circle.par(0)) - y0 * cos(circle.par(0)) + circle.par(1);
    uu = 1 + circle.par(2) * circle.par(1);
    C = -circle.par(2) * y0 + uu * cos(circle.par(0));
    BB = circle.par(2) * x0 + uu * sin(circle.par(0));
    A = 2. * DO + circle.par(2) * (Rfit::sqr(DO) + Rfit::sqr(DP));
    U = sqrt(1. + circle.par(2) * A);
    xi = 1. / (Rfit::sqr(BB) + Rfit::sqr(C));
    v = 1. + circle.par(2) * DO;
    lambda = (0.5 * A) / (U * Rfit::sqr(1. + U));
    mu = 1. / (U * (1. + U)) + circle.par(2) * lambda;
    zeta = Rfit::sqr(DO) + Rfit::sqr(DP);

    jacobian << xi * uu * v, -xi * Rfit::sqr(circle.par(2)) * DP, xi * DP, 2. * mu * uu * DP, 2. * mu * v,
        mu * zeta - lambda * A, 0, 0, 1.;

    circle.par(0) = atan2(BB, C);
    circle.par(1) = A / (1 + U);


    jacobiMult(jacobian, circle.cov, circle.cov, holder, tile);
  }

  /*!
    \brief Computes the data needed for the Broken Line fit procedure that are mainly common for the circle and the line fit.

    \param hits hits coordinates.
    \param hits_cov hits covariance matrix.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param B magnetic field in Gev/cm/c.
    \param results PreparedBrokenLineData to be filled (see description of PreparedBrokenLineData).
  */
  template <typename M3xN,
            typename V4,
            typename M2xN,
            int N,
            unsigned int TileSize>
  __device__ inline void prepareBrokenLineData(const M3xN& hits,
                                               const V4& fast_fit,
                                               const double B,
                                               PreparedBrokenLineData<N>& results,
                                               cg::thread_block_tile<TileSize>& tile,
                                               M2xN& pointsSZ) {
    constexpr auto n = N;

    u_int i = tile.thread_rank();
    Rfit::Vector2d d;
    Rfit::Vector2d e;

    d = hits.block(0, 1, 2, 1) - hits.block(0, 0, 2, 1);
    e = hits.block(0, n - 1, 2, 1) - hits.block(0, n - 2, 2, 1);
    results.q = Rfit::cross2D(d, e) > 0 ? -1 : 1;

    const double slope = -results.q / fast_fit(3);

    Rfit::Matrix2d R = RotationMatrix(slope);

    results.radii = hits.block(0, 0, 2, n) - fast_fit.head(2) * Rfit::MatrixXd::Constant(1, n, 1);
    e = -fast_fit(2) * fast_fit.head(2) / fast_fit.head(2).norm();

      d = results.radii.block(0, i, 2, 1);
      results.s(i) = results.q * fast_fit(2) * atan2(Rfit::cross2D(d, e), d.dot(e));  // calculates the arc length
    Rfit::VectorNd<N> z = hits.block(2, 0, 1, n).transpose();

    pointsSZ = Rfit::Matrix2xNd<N>::Zero();

      pointsSZ(0, i) = results.s(i);
      pointsSZ(1, i) = z(i);
      pointsSZ.block(0, i, 2, 1) = R * pointsSZ.block(0, i, 2, 1);  //TODO: cuBLASDx
    tile.sync();

    results.S = pointsSZ.block(0, 0, 1, n).transpose();
    results.Z = pointsSZ.block(1, 0, 1, n).transpose();

    results.VarBeta(0) = results.VarBeta(n - 1) = 0;

      results.VarBeta(i) = MultScatt(results.S(i + 1) - results.S(i), B, fast_fit(2), i + 2, slope) +
                           MultScatt(results.S(i) - results.S(i - 1), B, fast_fit(2), i + 1, slope);
    tile.sync();
  }

  /*!
    \brief Computes the n-by-n band matrix obtained minimizing the Broken Line's cost function w.r.t u. This is the whole matrix in the case of the line fit and the main n-by-n block in the case of the circle fit.

    \param w weights of the first part of the cost function, the one with the measurements and not the angles (\sum_{i=1}^n w*(y_i-u_i)^2).
    \param S total distance traveled by the particle from the pre-fitted closest approach.
    \param VarBeta kink angles' variance.

    \return the n-by-n matrix of the linear system
  */
  template <int N,
            typename VN,
            unsigned int TileSize,
            typename MNxN>
  __device__ inline void MatrixC_u(const VN& w,
                                   const Rfit::VectorNd<N>& S,
                                   const Rfit::VectorNd<N>& VarBeta,
                                   MNxN& C_U,
                                   cg::thread_block_tile<TileSize>& tile) {
    constexpr u_int n = N;
    u_int i = tile.thread_rank();

      C_U = Rfit::MatrixNd<N>::Zero();

      C_U(i, i) = w(i);
      if (i > 1)
        C_U(i, i) += 1. / (VarBeta(i - 1) * Rfit::sqr(S(i) - S(i - 1)));
      if (i > 0 && i < n - 1)
        C_U(i, i) += (1. / VarBeta(i)) * Rfit::sqr((S(i + 1) - S(i - 1)) / ((S(i + 1) - S(i)) * (S(i) - S(i - 1))));
      if (i < n - 2)
        C_U(i, i) += 1. / (VarBeta(i + 1) * Rfit::sqr(S(i + 1) - S(i)));

      if (i > 0 && i < n - 1)
        C_U(i, i + 1) =
            1. / (VarBeta(i) * (S(i + 1) - S(i))) * (-(S(i + 1) - S(i - 1)) / ((S(i + 1) - S(i)) * (S(i) - S(i - 1))));
      if (i < n - 2)
        C_U(i, i + 1) += 1. / (VarBeta(i + 1) * (S(i + 1) - S(i))) *
                         (-(S(i + 2) - S(i)) / ((S(i + 2) - S(i + 1)) * (S(i + 1) - S(i))));

      if (i < n - 2)
        C_U(i, i + 2) = 1. / (VarBeta(i + 1) * (S(i + 2) - S(i + 1)) * (S(i + 1) - S(i)));

      C_U(i, i) *= 0.5;


    //tile.sync();
    if (i == 0) {
      C_U += C_U.transpose();
    }
  }

  /*!
    \brief A very fast helix fit.

    \param hits the measured hits.

    \return (X0,Y0,R,tan(theta)).

    \warning sign of theta is (intentionally, for now) mistaken for negative charges.
  */

  template <typename M3xN, typename V4>
  __device__ inline void BL_Fast_fit(const M3xN& hits, V4& result) {
    constexpr uint32_t N = M3xN::ColsAtCompileTime;
    constexpr auto n = N;  // get the number of hits

    const Rfit::Vector2d a = hits.block(0, n / 2, 2, 1) - hits.block(0, 0, 2, 1);
    const Rfit::Vector2d b = hits.block(0, n - 1, 2, 1) - hits.block(0, n / 2, 2, 1);
    const Rfit::Vector2d c = hits.block(0, 0, 2, 1) - hits.block(0, n - 1, 2, 1);

    auto tmp = 0.5 / Rfit::cross2D(c, a);
    result(0) = hits(0, 0) - (a(1) * c.squaredNorm() + c(1) * a.squaredNorm()) * tmp;
    result(1) = hits(1, 0) + (a(0) * c.squaredNorm() + c(0) * a.squaredNorm()) * tmp;
    // check Wikipedia for these formulas

    result(2) = sqrt(a.squaredNorm() * b.squaredNorm() * c.squaredNorm()) / (2. * std::abs(Rfit::cross2D(b, a)));
    // Using Math Olympiad's formula R=abc/(4A)

    const Rfit::Vector2d d = hits.block(0, 0, 2, 1) - result.head(2);
    const Rfit::Vector2d e = hits.block(0, n - 1, 2, 1) - result.head(2);

    result(3) = result(2) * atan2(Rfit::cross2D(d, e), d.dot(e)) / (hits(2, n - 1) - hits(2, 0));
    // ds/dz slope between last and first point
  }

  /*!
    \brief Performs the Broken Line fit in the curved track case (that is, the fit parameters are the interceptions u and the curvature correction \Delta\kappa).

    \param hits hits coordinates.
    \param hits_cov hits covariance matrix.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param B magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineData.
    \param circle_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (phi, d, k); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.

    \details The function implements the steps 2 and 3 of the Broken Line fit with the curvature correction.\n
    The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and \Delta\kappa and their covariance matrix.
    The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
  */
  template <typename M3xN,
            typename M6xN,
            typename V4,
            typename VN,
            typename VNp1,
            typename MNp1,
            typename MNxN,
            typename M3x3,
            //int N,
            unsigned int TileSize>
  __device__ inline void BL_Circle_fit(const M3xN& hits,
                                       const M6xN& hits_ge,
                                       const V4& fast_fit,
                                       const double B,
                                       PreparedBrokenLineData<4>& data,
                                       karimaki_circle_fit& circle_results,
                                       VN& w,
                                       VNp1& r_u,
                                       MNp1& C_U,
                                       MNxN& C_UBlock,
                                       M3x3& jacobian,
                                       M3x3& holder,
                                       cg::thread_block_tile<TileSize>& tile) {
    u_int i = tile.thread_rank();

    circle_results.q = data.q;
    auto& radii = data.radii;
    const auto& s = data.s;
    const auto& S = data.S;
    auto& Z = data.Z;
    auto& VarBeta = data.VarBeta;
    const double slope = -circle_results.q / fast_fit(3);

    if (tile.thread_rank() == 0) {
      VarBeta *= 1. + Rfit::sqr(slope);  // the kink angles are projected!
    }

      Z(i) = radii.block(0, i, 2, 1).norm() - fast_fit(2);

      Rfit::Matrix2d V;   // covariance matrix
      Rfit::Matrix2d RR;  // rotation matrix point by point
                          //double Slope; // slope of the circle point by point

      V(0, 0) = hits_ge.col(i)[0];            // x errors
      V(0, 1) = V(1, 0) = hits_ge.col(i)[1];  // cov_xy
      V(1, 1) = hits_ge.col(i)[2];            // y errors
      RR = RotationMatrix(-radii(0, i) / radii(1, i));
      w(i) = 1. / ((RR * V * RR.transpose())(1, 1));  // compute the orthogonal weight point by point

    r_u(4) = 0;
    //tile.sync(); //FIXME: this might cause problems

      r_u(i) = w(i) * Z(i);


    MatrixC_u(w, s, VarBeta, C_UBlock, tile);


      C_U.block(0, 0, 4, 4) = C_UBlock;

      C_U(4, 4) = 0;

      C_U(i, 4) = 0;
      if (i > 0 && i < 4 - 1) {
        C_U(i, 4) +=
            -(s(i + 1) - s(i - 1)) * (s(i + 1) - s(i - 1)) / (2. * VarBeta(i) * (s(i + 1) - s(i)) * (s(i) - s(i - 1)));
      }
      if (i > 1) {
        C_U(i, 4) += (s(i) - s(i - 2)) / (2. * VarBeta(i - 1) * (s(i) - s(i - 1)));
      }
      if (i < 4 - 2) {
        C_U(i, 4) += (s(i + 2) - s(i)) / (2. * VarBeta(i + 1) * (s(i + 1) - s(i)));
      }
      C_U(4, i) = C_U(i, 4);
      if (i > 0 && i < 4 - 1) {
        double tmp3 = Rfit::sqr(s(i + 1) - s(i - 1)) / (4. * VarBeta(i));
        atomicAdd(&C_U(4, 4), tmp3);
    }

   tile.sync();

    Rfit::MatrixNplusONEd<4> I;
    math::cholesky::invert(C_U, I);


    Rfit::VectorNplusONEd<4> u = I * r_u;  // obtain the fitted parameters by solving the linear system
    if (tile.thread_rank() == 0) {
      // compute (phi, d_ca, k) in the system in which the midpoint of the first two corrected hits is the origin...

      radii.block(0, 0, 2, 1) /= radii.block(0, 0, 2, 1).norm();
      radii.block(0, 1, 2, 1) /= radii.block(0, 1, 2, 1).norm();
    }
    Rfit::Vector2d d = hits.block(0, 0, 2, 1) + (-Z(0) + u(0)) * radii.block(0, 0, 2, 1);
    Rfit::Vector2d e = hits.block(0, 1, 2, 1) + (-Z(1) + u(1)) * radii.block(0, 1, 2, 1);

      circle_results.par << atan2((e - d)(1), (e - d)(0)),
          -circle_results.q * (fast_fit(2) - sqrt(Rfit::sqr(fast_fit(2)) - 0.25 * (e - d).squaredNorm())),
          circle_results.q * (1. / fast_fit(2) + u(4));

      assert(circle_results.q * circle_results.par(1) <= 0);

      Rfit::Vector2d eMinusd = e - d;

      double tmp1 = eMinusd.squaredNorm();

      jacobian << (radii(1, 0) * eMinusd(0) - eMinusd(1) * radii(0, 0)) / tmp1,
          (radii(1, 1) * eMinusd(0) - eMinusd(1) * radii(0, 1)) / tmp1, 0,
          (circle_results.q / 2) * (eMinusd(0) * radii(0, 0) + eMinusd(1) * radii(1, 0)) /
              sqrt(Rfit::sqr(2 * fast_fit(2)) - tmp1),
          (circle_results.q / 2) * (eMinusd(0) * radii(0, 1) + eMinusd(1) * radii(1, 1)) /
              sqrt(Rfit::sqr(2 * fast_fit(2)) - tmp1),
          0, 0, 0, circle_results.q;

      circle_results.cov << I(0, 0), I(0, 1), I(0, 4), I(1, 0), I(1, 1), I(1, 4), I(4, 0), I(4, 1), I(4, 4);

    //tile.sync();

    jacobiMult(jacobian, circle_results.cov, circle_results.cov, holder, tile);

    //...Translate in the system in which the first corrected hit is the origin, adding the m.s. correction...

    TranslateKarimaki(circle_results, 0.5 * (e - d)(0), 0.5 * (e - d)(1), jacobian, holder, tile);  //TODO: cuBLASDx
    if (tile.thread_rank() == 0) {
      circle_results.cov(0, 0) +=
          (1 + Rfit::sqr(slope)) * MultScatt(S(1) - S(0), B, fast_fit(2), 2, slope);  //TODO: BE AWARE!!
    }
    //...And translate back to the original system

    TranslateKarimaki(circle_results, d(0), d(1), jacobian, holder, tile);  //TODO: cuBLASDx

    // compute chi2

    float tmp2 = 0;

      tmp2 = w(i) * Rfit::sqr(Z(i) - u(i));
      if (i > 0 && i < 4 - 1)
        tmp2 += Rfit::sqr(u(i - 1) / (s(i) - s(i - 1)) -
                          u(i) * (s(i + 1) - s(i - 1)) / ((s(i + 1) - s(i)) * (s(i) - s(i - 1))) +
                          u(i + 1) / (s(i + 1) - s(i)) + (s(i + 1) - s(i - 1)) * u(4) / 2) /
                VarBeta(i);

    cg::reduce_store_async(tile, &circle_results.chi2, tmp2, cg::plus<float>());

  }

  /*!
    \brief Performs the Broken Line fit in the straight track case (that is, the fit parameters are only the interceptions u).

    \param hits hits coordinates.
    \param hits_cov hits covariance matrix.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param B magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineData.
    \param line_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (cot(theta), Zip); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.

    \details The function implements the steps 2 and 3 of the Broken Line fit without the curvature correction.\n
    The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and their covariance matrix.
    The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
  */
  template <typename V4,
            typename M6xN,
            typename VN1,
            typename VN2,
            typename MNxN,
            typename M2x2,
            //int N,
            unsigned int TileSize>
  __device__ inline void BL_Line_fit(const M6xN& hits_ge,
                                     const V4& fast_fit,
                                     const double B,
                                     const PreparedBrokenLineData<4>& data,
                                     Rfit::line_fit& line_results,
                                     VN1& w,
                                     VN2& r_u,
                                     MNxN& C_U,
                                     M2x2& jacobian,
                                     M2x2& holder,
                                     cg::thread_block_tile<TileSize>& tile) {
    constexpr u_int n = 4;
    auto i = tile.thread_rank();

    const auto& radii = data.radii;
    const auto& S = data.S;
    const auto& Z = data.Z;
    const auto& VarBeta = data.VarBeta;

    const double slope = -data.q / fast_fit(3);
    Rfit::Matrix2d R = RotationMatrix(slope);

    Rfit::Matrix3d V = Rfit::Matrix3d::Zero();                 // covariance matrix XYZ
    Rfit::Matrix2x3d JacobXYZtosZ = Rfit::Matrix2x3d::Zero();  // jacobian for computation of the error on s (xyz -> sz)
    w = Rfit::VectorNd<4>::Zero();


      V(0, 0) = hits_ge.col(i)[0];            // x errors
      V(0, 1) = V(1, 0) = hits_ge.col(i)[1];  // cov_xy
      V(0, 2) = V(2, 0) = hits_ge.col(i)[3];  // cov_xz
      V(1, 1) = hits_ge.col(i)[2];            // y errors
      V(2, 1) = V(1, 2) = hits_ge.col(i)[4];  // cov_yz
      V(2, 2) = hits_ge.col(i)[5];            // z errors
      auto tmp = 1. / radii.block(0, i, 2, 1).norm();
      JacobXYZtosZ(0, 0) = radii(1, i) * tmp;
      JacobXYZtosZ(0, 1) = -radii(0, i) * tmp;
      JacobXYZtosZ(1, 2) = 1.;
      w(i) = 1. / ((R * JacobXYZtosZ * V * JacobXYZtosZ.transpose() * R.transpose())(  //TODO: cublasDx
                      1,
                      1));  // compute the orthogonal weight point by point


    //tile.sync(); // FIXME: this might cause problems

      r_u(i) = w(i) * Z(i);



    MatrixC_u(w, S, VarBeta, C_U, tile);

    Rfit::MatrixNd<4> I;

    tile.sync();
    math::cholesky::invert(C_U, I);


    Rfit::VectorNd<4> u = I * r_u;  // obtain the fitted parameters by solving the linear system

      // line parameters in the system in which the first hit is the origin and with axis along SZ
      line_results.par << (u(1) - u(0)) / (S(1) - S(0)), u(0);
      auto idiff = 1. / (S(1) - S(0));
      line_results.cov << (I(0, 0) - 2 * I(0, 1) + I(1, 1)) * Rfit::sqr(idiff) +
                              MultScatt(S(1) - S(0), B, fast_fit(2), 2, slope),
          (I(0, 1) - I(0, 0)) * idiff, (I(0, 1) - I(0, 0)) * idiff, I(0, 0);

      // translate to the original SZ system
      //Rfit::Matrix2d jacobian;

      jacobian(0, 0) = 1.;
      jacobian(0, 1) = 0;
      jacobian(1, 0) = -S(0);
      jacobian(1, 1) = 1.;

      if (tile.thread_rank() == 0) {
        line_results.par(1) += -line_results.par(0) * S(0);
      }
      //tile.sync();


      jacobiMult(jacobian, line_results.cov, line_results.cov, holder, tile);

        // rotate to the original sz system
        auto tmp1 = R(0, 0) - line_results.par(0) * R(0, 1);
        jacobian(1, 1) = 1. / tmp1;
        jacobian(0, 0) = jacobian(1, 1) * jacobian(1, 1);
        jacobian(0, 1) = 0;
        jacobian(1, 0) = line_results.par(1) * R(0, 1) * jacobian(0, 0);


        if (tile.thread_rank() == 0) {

          line_results.par(1) = line_results.par(1) * jacobian(1, 1);
          line_results.par(0) = (R(0, 1) + line_results.par(0) * R(0, 0)) * jacobian(1, 1);
        }
        //tile.sync();

        jacobiMult(jacobian, line_results.cov, line_results.cov, holder, tile);

        line_results.chi2 = 0;

        double tmp2 = 0;
        //if (i < N) {
          tmp2 = w(i) * Rfit::sqr(Z(i) - u(i));
          if (i > 0 && i < n - 1) {
            tmp2 += Rfit::sqr(u(i - 1) / (S(i) - S(i - 1)) -
                              u(i) * (S(i + 1) - S(i - 1)) / ((S(i + 1) - S(i)) * (S(i) - S(i - 1))) +
                              u(i + 1) / (S(i + 1) - S(i))) /
                    VarBeta(i);
          }

        cg::reduce_store_async(tile, &line_results.chi2, tmp2, cg::plus<double>());


      }

      /*!
    \brief Helix fit by three step:
    -fast pre-fit (see Fast_fit() for further info); \n
    -circle fit of the hits projected in the transverse plane by Broken Line algorithm (see BL_Circle_fit() for further info); \n
    -line fit of the hits projected on the (pre-fitted) cilinder surface by Broken Line algorithm (see BL_Line_fit() for further info); \n
    Points must be passed ordered (from inner to outer layer).

    \param hits Matrix3xNd hits coordinates in this form: \n
    |x1|x2|x3|...|xn| \n
    |y1|y2|y3|...|yn| \n
    |z1|z2|z3|...|zn|
    \param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n
    |(x1,x1)|(x2,x1)|(x3,x1)|(x4,x1)|.|(y1,x1)|(y2,x1)|(y3,x1)|(y4,x1)|.|(z1,x1)|(z2,x1)|(z3,x1)|(z4,x1)| \n
    |(x1,x2)|(x2,x2)|(x3,x2)|(x4,x2)|.|(y1,x2)|(y2,x2)|(y3,x2)|(y4,x2)|.|(z1,x2)|(z2,x2)|(z3,x2)|(z4,x2)| \n
    |(x1,x3)|(x2,x3)|(x3,x3)|(x4,x3)|.|(y1,x3)|(y2,x3)|(y3,x3)|(y4,x3)|.|(z1,x3)|(z2,x3)|(z3,x3)|(z4,x3)| \n
    |(x1,x4)|(x2,x4)|(x3,x4)|(x4,x4)|.|(y1,x4)|(y2,x4)|(y3,x4)|(y4,x4)|.|(z1,x4)|(z2,x4)|(z3,x4)|(z4,x4)| \n
    .       .       .       .       . .       .       .       .       . .       .       .       .       . \n
    |(x1,y1)|(x2,y1)|(x3,y1)|(x4,y1)|.|(y1,y1)|(y2,y1)|(y3,x1)|(y4,y1)|.|(z1,y1)|(z2,y1)|(z3,y1)|(z4,y1)| \n
    |(x1,y2)|(x2,y2)|(x3,y2)|(x4,y2)|.|(y1,y2)|(y2,y2)|(y3,x2)|(y4,y2)|.|(z1,y2)|(z2,y2)|(z3,y2)|(z4,y2)| \n
    |(x1,y3)|(x2,y3)|(x3,y3)|(x4,y3)|.|(y1,y3)|(y2,y3)|(y3,x3)|(y4,y3)|.|(z1,y3)|(z2,y3)|(z3,y3)|(z4,y3)| \n
    |(x1,y4)|(x2,y4)|(x3,y4)|(x4,y4)|.|(y1,y4)|(y2,y4)|(y3,x4)|(y4,y4)|.|(z1,y4)|(z2,y4)|(z3,y4)|(z4,y4)| \n
    .       .       .    .          . .       .       .       .       . .       .       .       .       . \n
    |(x1,z1)|(x2,z1)|(x3,z1)|(x4,z1)|.|(y1,z1)|(y2,z1)|(y3,z1)|(y4,z1)|.|(z1,z1)|(z2,z1)|(z3,z1)|(z4,z1)| \n
    |(x1,z2)|(x2,z2)|(x3,z2)|(x4,z2)|.|(y1,z2)|(y2,z2)|(y3,z2)|(y4,z2)|.|(z1,z2)|(z2,z2)|(z3,z2)|(z4,z2)| \n
    |(x1,z3)|(x2,z3)|(x3,z3)|(x4,z3)|.|(y1,z3)|(y2,z3)|(y3,z3)|(y4,z3)|.|(z1,z3)|(z2,z3)|(z3,z3)|(z4,z3)| \n
    |(x1,z4)|(x2,z4)|(x3,z4)|(x4,z4)|.|(y1,z4)|(y2,z4)|(y3,z4)|(y4,z4)|.|(z1,z4)|(z2,z4)|(z3,z4)|(z4,z4)|
    \param B magnetic field in the center of the detector in Gev/cm/c, in order to perform the p_t calculation.

    \warning see BL_Circle_fit(), BL_Line_fit() and Fast_fit() warnings.

    \bug see BL_Circle_fit(), BL_Line_fit() and Fast_fit() bugs.

    \return (phi,Tip,p_t,cot(theta)),Zip), their covariance matrix and the chi2's of the circle and line fits.
  */

    }  // namespace BrokenLine

#endif  // RecoPixelVertexing_PixelTrackFitting_interface_BrokenLineGPUq_h