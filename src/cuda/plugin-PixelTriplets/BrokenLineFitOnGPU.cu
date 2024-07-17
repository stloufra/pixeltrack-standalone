#include "BrokenLineFitOnGPU.h"
#include "CUDACore/device_unique_ptr.h"
#include "defs.h"
#include <chrono>
#include <iostream>
#include <cstdlib>

void HelixFitOnGPU::launchBrokenLineKernels(HitsView const *hv,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples,
                                            cudaStream_t stream) {
  assert(tuples_d);

  using time = std::chrono::high_resolution_clock;

  auto blockSize = 64; //if changed need to adjust shared memory
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  auto hitsGPU_ = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double), stream);
  auto hits_geGPU_ = cms::cuda::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float), stream);
  auto fast_fit_resultsGPU_ = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double), stream);

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernelBLFastFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 3, offset);
    cudaCheck(cudaGetLastError());
#ifdef __TIME__KERNELS__BROKENLINE
    auto startTrip = time::now();
#endif
    kernelBLFit<3><<<numberOfBlocks, __NUMBER_OF_BLOCKS*4, 0, stream>>>(tupleMultiplicity_d,
                                                             bField_,
                                                             outputSoa_d,
                                                             hitsGPU_.get(),
                                                             hits_geGPU_.get(),
                                                             fast_fit_resultsGPU_.get(),
                                                             3,
                                                             offset);
    cudaCheck(cudaGetLastError());
#ifdef __TIME__KERNELS__BROKENLINE
    auto endTrip = time::now();
#endif
    // fit quads
    kernelBLFastFit<4><<<numberOfBlocks /*/ 4*/, blockSize, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 4, offset);
    cudaCheck(cudaGetLastError());
#ifdef __TIME__KERNELS__BROKENLINE
    auto startQuad = time::now();
#endif

    kernelBLFit<4><<<numberOfBlocks /*/ 4*/, __NUMBER_OF_BLOCKS*4, 0, stream>>>(tupleMultiplicity_d,
                                                                 bField_,
                                                                 outputSoa_d,
                                                                 hitsGPU_.get(),
                                                                 hits_geGPU_.get(),
                                                                 fast_fit_resultsGPU_.get(),
                                                                 4,
                                                                 offset);
    cudaCheck(cudaGetLastError());
#ifdef __TIME__KERNELS__BROKENLINE
    auto endQuad = time::now();

    auto startPenta = time::now();
#endif
    if (fit5as4_) {
      // fit penta (only first 4)
      kernelBLFastFit<4><<<numberOfBlocks /*/ 4*/, blockSize, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(cudaGetLastError());



      kernelBLFit<4><<<numberOfBlocks /*/ 4*/, __NUMBER_OF_BLOCKS*4, 0, stream>>>(tupleMultiplicity_d,
                                                                   bField_,
                                                                   outputSoa_d,
                                                                   hitsGPU_.get(),
                                                                   hits_geGPU_.get(),
                                                                   fast_fit_resultsGPU_.get(),
                                                                   5,
                                                                   offset);
      cudaCheck(cudaGetLastError());


    } else {
      // fit penta (all 5)
      kernelBLFastFit<5><<<numberOfBlocks /*/ 4*/, blockSize, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(cudaGetLastError());


      kernelBLFit<5><<<numberOfBlocks /*/ 4*/, __NUMBER_OF_BLOCKS*8, 0, stream>>>(tupleMultiplicity_d,
                                                                   bField_,
                                                                   outputSoa_d,
                                                                   hitsGPU_.get(),
                                                                   hits_geGPU_.get(),
                                                                   fast_fit_resultsGPU_.get(),
                                                                   5,
                                                                   offset);
      cudaCheck(cudaGetLastError());


    }
#ifdef __TIME__KERNELS__BROKENLINE

    auto endPenta = time::now();


    globalTimeTriplets += endTrip - startTrip;
    globalTimeQuads += endQuad - startQuad;
    globalTimePenta += endPenta - startPenta;
#endif

  }  // loop on concurrent fits


}
