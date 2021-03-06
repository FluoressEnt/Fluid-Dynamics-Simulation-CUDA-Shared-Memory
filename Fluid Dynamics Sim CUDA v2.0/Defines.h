#include "cuda_runtime.h"
#include <iostream>
#include <cassert>

#pragma once

#ifndef DEFINES
#define DEFINES

#define RES 446										//resolution of the window  (needs to be divisible by BRES) working values:222:446
#define ALENGTH (RES+2)*(RES+2)						//total array size

#define VISC 0.005f									//viscosity constant
#define DT 0.004f									//static delta time
#define A DT * VISC * RES *RES						//conservation of diffusivity

#define BRES 16										//resolution of blocksize

#define TPBLOCK BRES*BRES							//threads per block
#define NBLOCK (ALENGTH + TPBLOCK - 1) / TPBLOCK	//number of blocks

#define CUDA_CHECK(fn) {\
                const cudaError_t rc = (fn);\
                if (rc != cudaSuccess) {\
                                std::cout << "CUDA Error: " << cudaGetErrorString(rc) << " (" << rc << ")" << std::endl;\
                                cudaDeviceReset();\
                                assert(0);\
                }\
}

#define CUDA_CHECK_POST() {\
                const cudaError_t rc = cudaGetLastError();\
                if (rc != cudaSuccess) {\
                                std::cout << "CUDA Error: " << cudaGetErrorString(rc) << " (" << rc << ")" << std::endl;\
                                cudaDeviceReset();\
                                assert(0);\
                }\
}

#endif