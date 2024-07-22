//
// Author: Felice Pantaleo, CERN
//

// #define BROKENLINE_DEBUG

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/cuda_assert.h"
#include "CondFormats/pixelCPEforGPU.h"
#include "defs.h"

#include "BrokenLineGPU.h"
#include "HelixFitOnGPU.h"

using HitsOnGPU = TrackingRecHit2DSOAView;
using Tuples = pixelTrack::HitContainer;
using OutputSoA = pixelTrack::TrackSoA;

// #define BL_DUMP_HITS

template <int N>
__global__ void kernelBLFastFit(Tuples const *__restrict__ foundNtuplets,
                                CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                HitsOnGPU const *__restrict__ hhp,
                                double *__restrict__ phits,
                                float *__restrict__ phits_ge,
                                double *__restrict__ pfast_fit,
                                uint32_t nHits,
                                uint32_t offset) {
  constexpr uint32_t hitsInFit = N;

  assert(hitsInFit <= nHits);

  assert(hhp);
  assert(pfast_fit);
  assert(foundNtuplets);
  assert(tupleMultiplicity);

  // look in bin for this hit multiplicity
  auto local_start = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef BROKENLINE_DEBUG
  if (0 == local_start) {
    printf("%d total Ntuple\n", foundNtuplets->nbins());
    printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
  }
#endif

  for (int local_idx = local_start, nt = Rfit::maxNumberOfConcurrentFits(); local_idx < nt;
       local_idx += gridDim.x * blockDim.x) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it from the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
    assert(tkid < foundNtuplets->nbins());

    assert(foundNtuplets->size(tkid) == nHits);

    Rfit::Map3xNd<N> hits(phits + local_idx);
    Rfit::Map4d fast_fit(pfast_fit + local_idx);
    Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);


#ifdef BL_DUMP_HITS
    __shared__ int done;
    done = 0;
    __syncthreads();
    bool dump = (foundNtuplets->size(tkid) == 5 && 0 == atomicAdd(&done, 1));
#endif

    // Prepare data structure
    auto const *hitId = foundNtuplets->begin(tkid);
    for (unsigned int i = 0; i < hitsInFit; ++i) {
      auto hit = hitId[i];
      float ge[6];
      hhp->cpeParams()
          .detParams(hhp->detectorIndex(hit))
          .frame.toGlobal(hhp->xerrLocal(hit), 0, hhp->yerrLocal(hit), ge);
#ifdef BL_DUMP_HITS
      if (dump) {
        printf("Hit global: %d: %d hits.col(%d) << %f,%f,%f\n",
               tkid,
               hhp->detectorIndex(hit),
               i,
               hhp->xGlobal(hit),
               hhp->yGlobal(hit),
               hhp->zGlobal(hit));
        printf("Error: %d: %d  hits_ge.col(%d) << %e,%e,%e,%e,%e,%e\n",
               tkid,
               hhp->detetectorIndex(hit),
               i,
               ge[0],
               ge[1],
               ge[2],
               ge[3],
               ge[4],
               ge[5]);
      }
#endif
      hits.col(i) << hhp->xGlobal(hit), hhp->yGlobal(hit), hhp->zGlobal(hit);
      hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
    }
    BrokenLine::BL_Fast_fit(hits, fast_fit);

    // no NaN here....
    assert(fast_fit(0) == fast_fit(0));
    assert(fast_fit(1) == fast_fit(1));
    assert(fast_fit(2) == fast_fit(2));
    assert(fast_fit(3) == fast_fit(3));
  }
}




template <int N>
__global__ void kernelBLFit(CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                            double B,
                            OutputSoA *results,
                            double *__restrict__ phits,
                            float *__restrict__ phits_ge,
                            double *__restrict__ pfast_fit,
                            uint32_t nHits,
                            uint32_t offset) {
  assert(N <= nHits);

  assert(results);
  assert(pfast_fit);

  // same as above...

  // number of threads in tile.
  constexpr unsigned int HitsToTileThreads[6] = {2, 2, 2, 4, 4, 8};  // Maximum of 5 hits for now
  const auto NumberOfThreadsPerTile = HitsToTileThreads[N];

  //cooperative group magic
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<NumberOfThreadsPerTile> tile = cg::tiled_partition<NumberOfThreadsPerTile>(block);

  // need to introduce const expresion for initialization of shared matrixes

  assert(__NUMBER_OF_BLOCKS == tile.meta_group_size());


  // look in bin for this hit multiplicity
  auto local_start = block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();

  for (int local_idx = local_start, nt = Rfit::maxNumberOfConcurrentFits(); local_idx < nt;
       local_idx += gridDim.x * tile.meta_group_size()) {


      auto tuple_idx = local_idx + offset;
      if (tuple_idx >= tupleMultiplicity->size(nHits))
        break;

      // get it for the ntuple container (one to one to helix)

      auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);

      //DATA PREP
#ifdef __BROKEN_LINE_WITH_SHARED_INPUTS
      Rfit::Map3xNd<N> Ghits(phits + local_idx);  //global memory map
      Rfit::Map4d Gfast_fit(pfast_fit + local_idx);
      Rfit::Map6xNf<N> Ghits_ge(phits_ge + local_idx);

      __shared__ Rfit::Matrix3xNd<N> hits[__NUMBER_OF_BLOCKS];  //shared memory
      __shared__ Eigen::Vector4d fast_fit[__NUMBER_OF_BLOCKS];
      __shared__ Rfit::Matrix6xNf<N> hits_ge[__NUMBER_OF_BLOCKS];


      auto tileId = tile.meta_group_rank();

      hits[tileId] = Ghits;
      fast_fit[tileId] = Gfast_fit;
      hits_ge[tileId] = Ghits_ge;
#else
      auto tileId = tile.meta_group_rank();

      Rfit::Map3xNd<N> hits(phits + local_idx);  //global memory map
      Rfit::Map4d fast_fit(pfast_fit + local_idx);
      Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);

#endif


      //structs for functions - prepare
      __shared__ Rfit::Matrix2xNd<N> pointsSZ[__NUMBER_OF_BLOCKS];

      //structs for functions - line fit
      __shared__ Rfit::VectorNd<N> w[__NUMBER_OF_BLOCKS]; //used for circle fit as well
      __shared__ Rfit::VectorNd<N> r_u[__NUMBER_OF_BLOCKS];
      __shared__ Rfit::MatrixNd<N> C_U[__NUMBER_OF_BLOCKS];//used for circle fit as well




    //structs for functions -  circle fit
      __shared__ Rfit::VectorNplusONEd<N> r_uc[__NUMBER_OF_BLOCKS];
      __shared__ Rfit::MatrixNplusONEd<N> C_Uc[__NUMBER_OF_BLOCKS];

#ifdef __BROKEN_LINE_WITH_SHARED_LINGEBRA
      __shared__ Rfit::Matrix3d jacobian3[__NUMBER_OF_BLOCKS];//used for circle fit as well
      __shared__ Rfit::Matrix3d holder3[__NUMBER_OF_BLOCKS];//used for circle fit as well

      __shared__ Rfit::Matrix2d jacobian2[__NUMBER_OF_BLOCKS];//used for circle fit as well
      __shared__ Rfit::Matrix2d holder2[__NUMBER_OF_BLOCKS];//used for circle fit as well
#else
      Rfit::Matrix3d jacobian3[__NUMBER_OF_BLOCKS];//used for circle fit as well
      Rfit::Matrix3d holder3[__NUMBER_OF_BLOCKS];//used for circle fit as well

      Rfit::Matrix2d jacobian2[__NUMBER_OF_BLOCKS];//used for circle fit as well
      Rfit::Matrix2d holder2[__NUMBER_OF_BLOCKS];//used for circle fit as well
#endif


      //PROCESS
#ifdef __BROKEN_LINE_WITH_SHARED_OUTPUTS
      __shared__ BrokenLine::PreparedBrokenLineData<N> data[__NUMBER_OF_BLOCKS];  //shared memory;

      __shared__ BrokenLine::karimaki_circle_fit circle[__NUMBER_OF_BLOCKS];  //shared memory;
      __shared__ Rfit::line_fit line[__NUMBER_OF_BLOCKS];                     //shared memory;
#else
      BrokenLine::PreparedBrokenLineData<N> data;

      BrokenLine::karimaki_circle_fit circle;
      Rfit::line_fit line;
#endif

#ifdef __BROKEN_LINE_WITH_SHARED_INPUTS

      BrokenLine::prepareBrokenLineData(hits[tileId], fast_fit[tileId], B, data[tileId], tile, pointsSZ[tileId]);

      BrokenLine::BL_Line_fit(hits_ge[tileId], fast_fit[tileId], B, data[tileId], line[tileId], w[tileId], r_u[tileId], C_U[tileId], jacobian2[tileId], holder2[tileId], tile);

      BrokenLine::BL_Circle_fit(hits[tileId], hits_ge[tileId], fast_fit[tileId], B, data[tileId], circle[tileId], w[tileId], r_uc[tileId], C_Uc[tileId], C_U[tileId], jacobian3[tileId], holder3[tileId], tile);

    if(tile.thread_rank() ==0) {
#else

      BrokenLine::prepareBrokenLineData(hits, fast_fit, B, data[tileId], tile, pointsSZ[tileId]);

      BrokenLine::BL_Line_fit(hits_ge, fast_fit, B, data[tileId], line, w[tileId], r_u[tileId], C_U[tileId], jacobian2, holder2[tileId], tile);

      BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit, B, data[tileId], circle, w[tileId], r_uc[tileId], C_Uc[tileId], C_U[tileId], jacobian3, holder3[tileId], tile);

    if(tile.thread_rank() ==0) {
#endif


      results->stateAtBS.copyFromCircle(circle[tileId].par, circle[tileId].cov, line[tileId].par, line[tileId].cov, 1.f / float(B), tkid);
      results->pt(tkid) = float(B) / float(std::abs(circle[tileId].par(2)));
      results->eta(tkid) = asinhf(line[tileId].par(0));
      results->chi2(tkid) = (circle[tileId].chi2 + line[tileId].chi2) / (2 * N - 5);

    }

#ifdef BROKENLINE_DEBUG
    if (!(circle.chi2 >= 0) || !(line.chi2 >= 0))
      printf("kernelBLFit failed! %f/%f\n", circle.chi2, line.chi2);
    printf("kernelBLFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
           N,
           nHits,
           tkid,
           circle.par(0),
           circle.par(1),
           circle.par(2));
    printf("kernelBLHits line.par(0,1): %d %f,%f\n", tkid, line.par(0), line.par(1));
    printf("kernelBLHits chi2 cov %f/%f  %e,%e,%e,%e,%e\n",
           circle.chi2,
           line.chi2,
           circle.cov(0, 0),
           circle.cov(1, 1),
           circle.cov(2, 2),
           line.cov(0, 0),
           line.cov(1, 1));
#endif
  }
}

