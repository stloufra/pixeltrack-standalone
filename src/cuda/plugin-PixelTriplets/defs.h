//
// Created by stloufra on 7/11/24.
//

#ifndef PIXELTRACK_STANDALONE_DEFS_H
#define PIXELTRACK_STANDALONE_DEFS_H

#include <chrono>

extern std::chrono::duration<double> globalTimeTriplets(0) ;
extern std::chrono::duration<double> globalTimeQuads(0) ;
extern std::chrono::duration<double> globalTimePenta(0) ;

// -------------DEFS - UNSAFE - JUST FOR NOW - !!!!!!!!!!!!

#define __BROKEN_LINE_WITH_SHARED_INPUTS
#define __BROKEN_LINE_WITH_SHARED_OUTPUTS
#define __BROKEN_LINE_WITH_SHARED_LINGEBRA

#define __BROKEN_LINE_ALL_THREADS


#define  __NUMBER_OF_BLOCKS  8

#define __TIME__KERNELS__BROKENLINE

// -------------DEFS - UNSAFE - JUST FOR NOW - !!!!!!!!!!!!

#endif  //PIXELTRACK_STANDALONE_DEFS_H
