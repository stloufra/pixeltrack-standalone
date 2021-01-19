#include "hip/hip_runtime.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>

#include "CUDACore/HistoContainer.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/launch.h"
#include "CUDACore/requireDevices.h"

using namespace cms::hip;

template <typename T, int NBINS, int S, int DELTA>
__global__ void mykernel(T const* __restrict__ v, uint32_t N) {
  assert(v);
  assert(N == 12000);

  if (threadIdx.x == 0)
    printf("start kernel for %d data\n", N);

  using Hist = HistoContainer<T, NBINS, 12000, S, uint16_t>;

  __shared__ Hist hist;
  __shared__ typename Hist::Counter ws[warpSize];

  for (uint32_t j = threadIdx.x; j < Hist::totbins(); j += static_cast<uint32_t>(blockDim.x)) {
    hist.off[j] = 0;
  }
  __syncthreads();

  for (uint32_t j = threadIdx.x; j < N; j += static_cast<uint32_t>(blockDim.x))
    hist.count(v[j]);
  __syncthreads();

  assert(0 == hist.size());
  __syncthreads();

  hist.finalize(ws);
  __syncthreads();

  assert(N == hist.size());
  for (uint32_t j = threadIdx.x; j < Hist::nbins(); j += static_cast<uint32_t>(blockDim.x))
    assert(hist.off[j] <= hist.off[j + 1]);
  __syncthreads();

  if (threadIdx.x < warpSize)
    ws[threadIdx.x] = 0;  // used by prefix scan...
  __syncthreads();

  for (uint32_t j = threadIdx.x; j < N; j += static_cast<uint32_t>(blockDim.x))
    hist.fill(v[j], j);
  __syncthreads();
  assert(0 == hist.off[0]);
  assert(N == hist.size());

  for (uint32_t j = threadIdx.x; j < hist.size() - 1; j += static_cast<uint32_t>(blockDim.x)) {
    auto p = hist.begin() + j;
    assert((*p) < N);
    auto k1 = Hist::bin(v[*p]);
    auto k2 = Hist::bin(v[*(p + 1)]);
    assert(k2 >= k1);
  }

  for (uint32_t i = threadIdx.x; i < hist.size(); i += static_cast<uint32_t>(blockDim.x)) {
    auto p = hist.begin() + i;
    auto j = *p;
    auto b0 = Hist::bin(v[j]);
    int tot = 0;
    auto ftest = [&](uint32_t k) {
      assert(k >= 0 && k < N);
      ++tot;
    };
    forEachInWindow(hist, v[j], v[j], ftest);
    int rtot = hist.size(b0);
    assert(tot == rtot);
    tot = 0;
    auto vm = int(v[j]) - DELTA;
    auto vp = int(v[j]) + DELTA;
    constexpr int vmax = NBINS != 128 ? NBINS * 2 - 1 : std::numeric_limits<T>::max();
    vm = std::max(vm, 0);
    vm = std::min(vm, vmax);
    vp = std::min(vp, vmax);
    vp = std::max(vp, 0);
    assert(vp >= vm);
    forEachInWindow(hist, vm, vp, ftest);
    int bp = Hist::bin(vp);
    int bm = Hist::bin(vm);
    rtot = hist.end(bp) - hist.begin(bm);
    assert(tot == rtot);
  }
}

template <typename T, int NBINS = 128, int S = 8 * sizeof(T), int DELTA = 1000>
void go() {
  std::mt19937 eng;

  int rmin = std::numeric_limits<T>::min();
  int rmax = std::numeric_limits<T>::max();
  if (NBINS != 128) {
    rmin = 0;
    rmax = NBINS * 2 - 1;
  }

  std::uniform_int_distribution<T> rgen(rmin, rmax);

  constexpr int N = 12000;
  T v[N];

  auto v_d = make_device_unique<T[]>(N, nullptr);
  assert(v_d.get());

  using Hist = HistoContainer<T, NBINS, N, S>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::capacity() << ' '
            << (rmax - rmin) / Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' << int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;

  for (int it = 0; it < 5; ++it) {
    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);
    if (it == 2)
      for (long long j = N / 2; j < N / 2 + N / 4; j++)
        v[j] = 4;

    assert(v_d.get());
    assert(v);
    cudaCheck(hipMemcpy(v_d.get(), v, N * sizeof(T), hipMemcpyHostToDevice));
    assert(v_d.get());
    launch(mykernel<T, NBINS, S, DELTA>, {1, 256}, v_d.get(), N);
  }
}

int main() {
  cms::hiptest::requireDevices();

  go<int16_t>();
  go<uint8_t, 128, 8, 4>();
  go<uint16_t, 313 / 2, 9, 4>();

  return 0;
}