//
// Created by stloufra on 7/11/24.
//

#ifndef PIXELTRACK_STANDALONE_DEFS_H
#define PIXELTRACK_STANDALONE_DEFS_H

#include <chrono>

// -------------DEFS - UNSAFE - JUST FOR NOW - !!!!!!!!!!!!

//Memory (switch only inputs)
#define __BROKEN_LINE_WITH_SHARED_INPUTS
#define __BROKEN_LINE_WITH_SHARED_OUTPUTS
#define __BROKEN_LINE_WITH_SHARED_LINGEBRA


#define __BROKEN_LINE_ALL_THREADS
#define __GROUPS_PER_BLOCK 8

//time
//#define __TIME__KERNELS__BROKENLINE

#ifdef __TIME__KERNELS__BROKENLINE
extern std::chrono::duration<double> globalTimeTriplets(0) ;
extern std::chrono::duration<double> globalTimeQuads(0) ;
extern std::chrono::duration<double> globalTimePenta(0) ;
#endif


//Multiplication (only one)
//#define __MULTIPLY_ONE_STEP_PARALLEL
#define __MULTIPLY_MULTIPLE_STEPS_PARALLEL
//#define __MULTIPLY_SERIAL

//IFS
#define __IFS_FOR_0_THREAD

//layout
//#define __NOT_COALESED_LAYOUT__
#define __COALESED_LAYOUT__


// -------------DEFS - UNSAFE - JUST FOR NOW - !!!!!!!!!!!!

#endif  //PIXELTRACK_STANDALONE_DEFS_H
