#ifndef PIXELTRACK_STANDALONE_MATMULABA_H
#define PIXELTRACK_STANDALONE_MATMULABA_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cooperative_groups.h>
#include <cmath>



//#ifdef __CUDACC__
namespace math {
  namespace squareMatrix {

    namespace cg = cooperative_groups;
      //--------------------------------------------------------------------------------------

      /*template <typename MATRIX, typename TILE>
      inline  __device__ void multABAt11(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
        if (tile.thread_rank() == 0) {
          C(0, 0) = A(0, 0) * B(0, 0) * A(0, 0);
        }
      }

      template <typename MATRIX, typename TILE>
      inline  __device__ void multABAt22(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
        auto idx = tile.thread_rank();
         MATRIX holder;

         unsigned int N = 2;

        for (auto i = idx; i < N * N; i += tile.num_threads()) {
          auto row = i / N;
          auto col = i % N;

          double tmp;

          for (unsigned int k = 0; k < N; k++) {
            for (unsigned int m = 0; m < N; k++)
              tmp += A(row, k) * B(k, m) * A(col, m);
          }

          holder(row, col) = tmp;
        }
      }
      */
      template <typename MATRIX, typename TILE>
      inline  __device__ void multABAt33(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
        auto idx = tile.thread_rank();
         Eigen::Matrix<double, 3, 3> holder;

        //MATRIX holder;

         unsigned int N = 3;

        for (auto i = idx; i < N * N; i += tile.num_threads()) {
          auto row = i / N;
          auto col = i % N;

          double tmp;

          for (unsigned int k = 0; k < N; k++) {
            for (unsigned int m = 0; m < N; m++)
              tmp += A(row, k) * B(k, m) * A(col, m);
          }

          holder(row, col) = tmp;
        }

        C = holder;
      }
      /*
      template <typename MATRIX, typename TILE>
      inline  __device__ void multABAt44(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
        auto idx = tile.thread_rank();
         MATRIX holder;

         unsigned int N = 4;

        for (auto i = idx; i < N * N; i += tile.num_threads()) {
          auto row = i / N;
          auto col = i % N;

          double tmp;

          for (unsigned int k = 0; k < N; k++) {
            for (unsigned int m = 0; m < N; k++)
              tmp += A(row, k) * B(k, m) * A(col, m);
          }

          holder(row, col) = tmp;
        }
      }

      template <typename MATRIX, typename TILE>
      inline  __device__ void multABAt55(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
        auto idx = tile.thread_rank();
         MATRIX holder;

         unsigned int N = 5;

        for (auto i = idx; i < N * N; i += tile.num_threads()) {
          auto row = i / N;
          auto col = i % N;

          double tmp;

          for (unsigned int k = 0; k < N; k++) {
            for (unsigned int m = 0; m < N; k++)
              tmp += A(row, k) * B(k, m) * A(col, m);
          }

          holder(row, col) = tmp;
        }
      }

      template <typename MATRIX, typename TILE>
      inline  __device__ void multABAt66(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
        auto idx = tile.thread_rank();
         MATRIX holder;
        

         unsigned int N = 6;

        for (auto i = idx; i < N * N; i += tile.num_threads()) {
          auto row = i / N;
          auto col = i % N;

          double tmp;

          for (unsigned int k = 0; k < N; k++) {
            for (unsigned int m = 0; m < N; k++)
              tmp += A(row, k) * B(k, m) * A(col, m);
          }

          holder(row, col) = tmp;
        }
      }
      */
      template <typename MATRIX, typename TILE, int N>
      struct MultiplierABAtEqC {
        static  __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { C = A * B * A.transpose(); }
      };
      /*
      template <typename MATRIX, typename TILE>
      struct MultiplierABAtEqC<MATRIX, TILE, 1> {
        static  __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multABAt11(A, B, C, tile); }
      };

      template <typename MATRIX, typename TILE>
      struct MultiplierABAtEqC<MATRIX, TILE, 2> {
        static  __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multABAt22(A, B, C, tile); }
      };
      */
      template <typename MATRIX, typename TILE>
      struct MultiplierABAtEqC<MATRIX, TILE, 3> {
        static  __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multABAt33(A, B, C, tile); }
      };
      /*
      template <typename MATRIX, typename TILE>
      struct MultiplierABAtEqC<MATRIX, TILE, 4> {
        static  __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multABAt44(A, B, C, tile); }
      };

      template <typename MATRIX, typename TILE>
      struct MultiplierABAtEqC<MATRIX, TILE, 5> {
        static  __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multABAt55(A, B, C, tile); }
      };

      template <typename MATRIX, typename TILE>
      struct MultiplierABAtEqC<MATRIX, TILE, 6> {
        static  __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multABAt66(A, B, C, tile); }
      };
      */
      // Eigen interface
      template <typename DENSE, unsigned int TileSize>
      inline  __device__ void multiplyABAtEqC(Eigen::DenseBase<DENSE> const& A,
                                            Eigen::DenseBase<DENSE> const& B,
                                            Eigen::DenseBase<DENSE>& C,
                                            cg::thread_block_tile<TileSize>& tile) {
        using MATRIX = Eigen::DenseBase<DENSE>;
        using TILE = cg::thread_block_tile<TileSize>;

        MultiplierABAtEqC<MATRIX, TILE, MATRIX::ColsAtCompileTime>::eval(A, B, C, tile);
      }
      }  // namespace squareMatrix
    }  // namespace math

#endif  //PIXELTRACK_STANDALONE_MATMULABA_H
