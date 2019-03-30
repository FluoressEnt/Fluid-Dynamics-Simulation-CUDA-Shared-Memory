#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "Solver.h"

void Solver::CalculateWrapper() {
	//global memory block and grid declarations
	dim3 gBlockSize(BRES, BRES);
	dim3 gGridSize = dim3((RES + gBlockSize.x + 2) / gBlockSize.x, (RES + gBlockSize.y + 2) / gBlockSize.y);

	//Shared memory block and grid declarations
	dim3 sBlockSize(BRES + 2, BRES + 2);
	dim3 sGridSize = dim3((RES + sBlockSize.x) / (sBlockSize.x - 2), (RES + sBlockSize.y) / (sBlockSize.y - 2));

	//diffusion part
	cAddSourceK << <gGridSize, gBlockSize >> > (cSDens, cNewDens, cOldDens);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcDiffusion << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewDens, cOldDens);
		cCalcBound << <gGridSize, gBlockSize >> > (1, cNewDens);
	}
	cCalcAdvection << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelX, cNewVelY, cNewDens, cOldDens);

	//velocity part
	cAddSourceK << <gGridSize, gBlockSize >> > (cSVelX, cNewVelX, cOldVelX);
	cAddSourceK << <gGridSize, gBlockSize >> > (cSVelY, cNewVelY, cOldVelY);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcDiffusion << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelX, cOldVelX);
		cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
	}
	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcDiffusion << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelY, cOldVelY);
		cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);
	}

	cCalcProjY << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelX, cNewVelY, cOldVelX, cOldVelY);
	cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelY);
	cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcProjX << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cOldVelX, cOldVelY);
		cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);
	}

	cCalcFinalProj << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelX, cNewVelY, cOldVelX);
	cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
	cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);

	cSwapPtr << <gGridSize, gBlockSize >> > (cOldVelX, cNewVelX);
	cSwapPtr << <gGridSize, gBlockSize >> > (cOldVelY, cNewVelY);

	cCalcAdvection << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelX, cOldVelX, cOldVelX, cOldVelY);
	cCalcAdvection << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelY, cOldVelY, cOldVelX, cOldVelY);

	cCalcProjY << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelX, cNewVelY, cOldVelX, cOldVelY);
	cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelY);
	cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcProjX << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cOldVelX, cOldVelY);
		cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);
	}

	cCalcFinalProj << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewVelX, cNewVelY, cOldVelX);
	cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
	cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);
}

//global memory kernel
__global__ void cAddSourceK(float* inputArr, float* arrayAffected, float* arrayToSwap) {
	cAddSource(arrayAffected, inputArr);
	__syncthreads();

	cSwap(&arrayToSwap, &arrayAffected);
	__syncthreads();
}
//shared memory kernel
__global__ void cCalcDiffusion(float* cNewDens, float* cOldDens) {
	cDiffuse(cNewDens, cOldDens);
	__syncthreads();
}
//shared memory kernel
__global__ void cCalcAdvection(float* cNewVelX, float* cNewVelY, float* cNewDens, float* cOldDens) {
	cSwap(&cOldDens, &cNewDens);
	__syncthreads();

	cAdvection(cOldDens, cNewDens, cNewVelX, cNewVelY);
	__syncthreads();
}
//shared memory kernel
__global__ void cCalcProjY(float* cNewVelX, float* cNewVelY, float* cOldVelX, float* cOldVelY) {
	cProjectionInY(cNewVelX, cNewVelY, cOldVelX, cOldVelY);
}
//shared memory kernel
__global__ void cCalcProjX(float* cOldVelX, float* cOldVelY) {
	cProjectionInX(cOldVelX, cOldVelY);
}
//shared memory kernel
__global__ void cCalcFinalProj(float* cNewVelX, float* cNewVelY, float* cOldVelX) {
	cFinalProjection(cNewVelX, cNewVelY, cOldVelX);
}

//global memory kernel
__global__ void cCalcBound(int b, float* cNewDens) {
	cSetBound(b, cNewDens);
}

//global memory kernel
__global__ void cSwapPtr(float* arrayOne, float* arrayTwo) {
	cSwap(&arrayOne, &arrayTwo);
	__syncthreads();
}

//global memory
__device__ void cAddSource(float* cNewDens, float *sourceArray) {
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = xPos + (RES + 2)*yPos;

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		float temp;
		temp = (DT * sourceArray[ID]);
		cNewDens[ID] += temp;
	}
}

//shared memory
__device__ void cDiffuse(float* cNewDens, float* cOldDens) {
	int col = threadIdx.y;
	int row = threadIdx.x;

	int blockStart = blockIdx.x * (blockDim.x - 2) + (RES + 2) * blockIdx.y * (blockDim.y - 2);
	int ID = blockStart + threadIdx.y * (RES + 2) + threadIdx.x;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sNewDens[BRES + 2][BRES + 2];
	__shared__ float sOldDens[BRES + 2][BRES + 2];

	sNewDens[col][row] = cNewDens[ID];
	sOldDens[col][row] = cOldDens[ID];

	__syncthreads();

	//ensures a boundary is maintained around the block where only reads occur
	if ((row > 0) && (row < BRES + 1) && (col > 0) && (col < BRES + 1)) {

		cNewDens[ID] = (sOldDens[col][row] + A * (sNewDens[col][row - 1] + sNewDens[col][row + 1]
			+ sNewDens[col - 1][row] + sNewDens[col + 1][row])) / (1 + 4 * A);
	}
}
//shared memory
__device__ void cAdvection(float* cNewDens, float* cOldDens, float* cVelX, float* cVelY) {
	int left, bottom, right, top;
	float x, y, distToRight, distToTop, distToLeft, distToBottom;

	int col = threadIdx.y;
	int row = threadIdx.x;

	int blockStart = blockIdx.x * (blockDim.x - 2) + (RES + 2) * blockIdx.y * (blockDim.y - 2);
	int ID = blockStart + threadIdx.y * (RES + 2) + threadIdx.x;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sOldDens[BRES + 2][BRES + 2];
	sOldDens[col][row] = cOldDens[ID];
	__syncthreads();

	if ((row > 0) && (row < BRES + 1) && (col > 0) && (col < BRES + 1)) {

		x = row - DT * RES * cVelX[ID];
		y = col - DT * RES * cVelY[ID];

		//neighbourhood of previous position
		if (x < 0.5) x = 0.5f;
		if (x > RES + 0.5) x = RES + 0.5f;
		left = (int)x;
		right = left + 1;

		if (y < 0.5) y = 0.5;
		if (y > RES + 0.5) y = RES + 0.5f;
		bottom = (int)y;
		top = bottom + 1;

		//interpolation part
		distToLeft = x - left;
		distToRight = 1 - distToLeft;
		distToBottom = y - bottom;
		distToTop = 1 - distToBottom;

		cNewDens[ID] = distToRight * (distToTop * sOldDens[bottom][left] + distToBottom * sOldDens[top][left])
			+ distToLeft * (distToTop * sOldDens[bottom][right] + distToBottom * sOldDens[top][right]);
	}
}
//shared memory
__device__ void cFinalProjection(float* cNewVelX, float* cNewVelY, float* cOldVelX) {
	int col = threadIdx.y;
	int row = threadIdx.x;

	int blockStart = blockIdx.x * (blockDim.x - 2) + (RES + 2) * blockIdx.y * (blockDim.y - 2);
	int ID = blockStart + threadIdx.y * (RES + 2) + threadIdx.x;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sOldVelX[BRES + 2][BRES + 2];
	sOldVelX[col][row] = cOldVelX[ID];
	__syncthreads();

	float h = 1.0f / RES;

	if ((row > 0) && (row < BRES + 1) && (col> 0) && (col < BRES + 1)) {
		cNewVelX[ID] -= 0.5f*(sOldVelX[col][row + 1] - sOldVelX[col][row - 1]) / h;
		cNewVelY[ID] -= 0.5f*(sOldVelX[col + 1][row] - sOldVelX[col - 1][row]) / h;
	}
}
//shared memory
__device__ void cProjectionInY(float* cNewVelX, float* cNewVelY, float* cOldVelX, float* cOldVelY) {
	int col = threadIdx.y;
	int row = threadIdx.x;

	int blockStart = blockIdx.x * (blockDim.x - 2) + (RES + 2) * blockIdx.y * (blockDim.y - 2);
	int ID = blockStart + threadIdx.y * (RES + 2) + threadIdx.x;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sNewVelX[BRES + 2][BRES + 2];
	__shared__ float sNewVelY[BRES + 2][BRES + 2];

	sNewVelX[col][row] = cNewVelX[ID];
	sNewVelY[col][row] = cNewVelY[ID];
	__syncthreads();

	float h = 1.0f / RES;

	if ((row > 0) && (row < BRES + 1) && (col > 0) && (col <= BRES + 1)) {
		cOldVelY[ID] = -0.5f * h * (sNewVelX[col][row + 1] - sNewVelX[col][row - 1]
			+ sNewVelY[col + 1][row] - sNewVelY[col - 1][row]);
		cOldVelX[ID] = 0;
	}
}
//shared memory
__device__ void cProjectionInX(float* cOldVelX, float* cOldVelY) {
	int col = threadIdx.y;
	int row = threadIdx.x;

	int blockStart = blockIdx.x * (blockDim.x - 2) + (RES + 2) * blockIdx.y * (blockDim.y - 2);
	int ID = blockStart + threadIdx.y * (RES + 2) + threadIdx.x;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sOldVelX[BRES + 2][BRES + 2];
	__shared__ float sOldVelY[BRES + 2][BRES + 2];

	sOldVelX[col][row] = cOldVelX[ID];
	sOldVelY[col][row] = cOldVelY[ID];
	__syncthreads();

	if ((row > 0) && (row < BRES + 1) && (col > 0) && (col < BRES + 1)) {
		cOldVelX[ID] = (sOldVelY[col][row] + sOldVelX[col][row - 1]
			+ sOldVelX[col][row + 1] + sOldVelX[col - 1][row] + sOldVelX[col + 1][row]) / 4;
	}
}

//global memory
__device__ void cSetBound(int b, float* boundArray) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int collumn = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = row + (RES + 2) * collumn;

	int xPos, yPos;
	xPos = cGetX(ID);
	yPos = cGetY(ID);

	if (xPos > 0 && xPos <= RES && yPos == 0) {
		boundArray[ID] = b == 2 ? -boundArray[ID + RES + 2] : boundArray[ID + RES + 2];
	}
	else if (xPos > 0 && xPos <= RES && yPos == RES + 1) {
		boundArray[ID] = b == 2 ? -boundArray[ID - RES - 2] : boundArray[ID - RES - 2];
	}
	else if (xPos == 1 && yPos > 0 && yPos <= RES) {
		boundArray[ID] = b == 1 ? -boundArray[ID + 1] : boundArray[ID + 1];
	}
	else if (xPos == RES + 1 && yPos > 0 && yPos <= RES) {
		boundArray[ID] = b == 1 ? -boundArray[ID - 1] : boundArray[ID - 1];
	}

	else if (xPos == 0 && yPos == 0) {
		boundArray[ID] = 0.5f *(boundArray[ID + 1] + boundArray[ID + RES + 2]);
	}
	else if (xPos == 0 && yPos == RES + 1) {
		boundArray[ID] = 0.5f *(boundArray[ID + 1] + boundArray[ID - RES - 2]);
	}
	else if (xPos == RES + 1 && yPos == 0) {
		boundArray[ID] = 0.5f *(boundArray[ID - 1] + boundArray[ID + RES + 2]);
	}
	else if (xPos == RES + 1 && yPos == RES + 1) {
		boundArray[ID] = 0.5f *(boundArray[ID - 1] + boundArray[ID - RES - 2]);
	}
}

__device__ void cSwap(float** arrayOne, float** arrayTwo) {
	int collumn = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = row * (RES + 2) + collumn;

	if (ID == 0) {
		float* temp = *arrayOne;
		*arrayOne = *arrayTwo;
		*arrayTwo = temp;
	}
}

__device__ int cGetX(int arrayPos) {
	return (arrayPos % (RES + 2));
}

__device__ int cGetY(int arrayPos) {
	return (arrayPos / (RES + 2));
}

__device__ int cGetArrayPos(int xPos, int yPos) {
	return xPos + (RES + 2)*yPos;
}
