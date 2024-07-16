#ifndef PIXELTRACK_STANDALONE_MATMUL_H
#define PIXELTRACK_STANDALONE_MATMUL_H

#include <Eigen/Core>
#include <cooperative_groups.h>
#include <cmath>

//#ifdef __CUDACC__
namespace math {
  namespace squareMatrix {

    namespace cg = cooperative_groups;

    template <typename MATRIX, typename TILE>
    inline __device__ void multAB11(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
      if (tile.thread_rank() == 0) {
        C(0, 0) = A(0, 0) * B(0, 0);
      }
    }

    template <typename MATRIX, typename TILE, typename T>
    inline __device__ void multAB22(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
      auto idx = tile.thread_rank();
      __shared__ MATRIX holder;

      unsigned int N = 2;

      for (auto i = idx; i < N * N; i += tile.num_threads()) {
        auto row = i / N;
        auto col = i % N;

        T tmp;
        for (unsigned int k = 0; k < N; k++) {
          tmp += A(row, k) * B(k, col);
        }

        holder(row, col) = tmp;
      }
    }

    template <typename MATRIX, typename TILE, typename T>
    inline __device__ void multAB33(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
      auto idx = tile.thread_rank();
      __shared__ MATRIX holder;

      unsigned int N = 3;

      for (auto i = idx; i < N * N; i += tile.num_threads()) {
        auto row = i / N;
        auto col = i % N;

        T tmp;
        for (unsigned int k = 0; k < N; k++) {
          tmp += A(row, k) * B(k, col);
        }

        holder(row, col) = tmp;
      }
    }

    template <typename MATRIX, typename TILE, typename T>
    inline __device__ void multAB44(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
      auto idx = tile.thread_rank();
      __shared__ MATRIX holder;

      unsigned int N = 4;

      for (auto i = idx; i < N * N; i += tile.num_threads()) {
        auto row = i / N;
        auto col = i % N;

        T tmp;
        for (unsigned int k = 0; k < N; k++) {
          tmp += A(row, k) * B(k, col);
        }

        holder(row, col) = tmp;
      }
    }

    template <typename MATRIX, typename TILE, typename T>
    inline __device__ void multAB55(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
      auto idx = tile.thread_rank();
      __shared__ MATRIX holder;

      unsigned int N = 5;

      for (auto i = idx; i < N * N; i += tile.num_threads()) {
        auto row = i / N;
        auto col = i % N;

        T tmp;
        for (unsigned int k = 0; k < N; k++) {
          tmp += A(row, k) * B(k, col);
        }

        holder(row, col) = tmp;
      }
    }

    template <typename MATRIX, typename TILE, typename T>
    inline __device__ void multAB66(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) {
      auto idx = tile.thread_rank();
      __shared__ MATRIX holder;

      unsigned int N = 6;

      for (auto i = idx; i < N * N; i += tile.num_threads()) {
        auto row = i / N;
        auto col = i % N;

        T tmp;
        for (unsigned int k = 0; k < N; k++) {
          tmp += A(row, k) * B(k, col);
        }

        holder(row, col) = tmp;
      }
    }

    template <typename MATRIX, typename TILE, int N>
    struct MultiplierABeqC {
      static __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { C = A * B; }
    };

    template <typename MATRIX, typename TILE>
    struct MultiplierABeqC<MATRIX, TILE, 1> {
      static __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multAB11(A, B, C, tile); }
    };

    template <typename MATRIX, typename TILE>
    struct MultiplierABeqC<MATRIX, TILE, 2> {
      static __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multAB22(A, B, C, tile); }
    };

    template <typename MATRIX, typename TILE>
    struct MultiplierABeqC<MATRIX, TILE, 3> {
      static __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multAB33(A, B, C, tile); }
    };

    template <typename MATRIX, typename TILE>
    struct MultiplierABeqC<MATRIX, TILE, 4> {
      static __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multAB44(A, B, C, tile); }
    };

    template <typename MATRIX, typename TILE>
    struct MultiplierABeqC<MATRIX, TILE, 5> {
      static __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multAB55(A, B, C, tile); }
    };

    template <typename MATRIX, typename TILE>
    struct MultiplierABeqC<MATRIX, TILE, 6> {
      static __device__ void eval(MATRIX const& A, MATRIX const& B, MATRIX& C, TILE& tile) { multAB66(A, B, C, tile); }
    };

    // Eigen interface
    template <typename DENSE, unsigned int TileSize>
    inline __device__ void multiplyABeqC(Eigen::DenseBase<DENSE> const& A,
                                         Eigen::DenseBase<DENSE> const& B,
                                         Eigen::DenseBase<DENSE>& C,
                                         cg::thread_block_tile<TileSize>& tile) {
      using MATRIX = Eigen::DenseBase<DENSE>;
      using TILE = cg::thread_block_tile<TileSize>;

      MultiplierABeqC<MATRIX, TILE, MATRIX::ColsAtCompileTime>::eval(A, B, C, tile);
    }

  }  // namespace squareMatrix
}  // namespace math

#endif  //PIXELTRACK_STANDALONE_MATMUL_H
