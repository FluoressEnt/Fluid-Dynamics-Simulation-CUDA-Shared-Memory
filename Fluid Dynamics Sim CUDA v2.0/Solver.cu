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

	//global memory kernel
	cAddSourceKernel << <gGridSize, gBlockSize >> > (cSDens, cNewDens, cOldDens);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		//shared memory kernel
		cCalcDens << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cNewDens, cOldDens);
		//global memory kernel
		cCalcBound << <gGridSize, gBlockSize >> > (cNewDens);
	}
	//shared memory kernal
	cCalcAdv << <sGridSize, sBlockSize, (sBlockSize.x)*(sBlockSize.y) * sizeof(float) >> > (cSVelX, cSVelY, cNewDens, cOldDens);
}

__global__ void cAddSourceKernel(float* cSDens, float* cNewDens, float* cOldDens) {
	cAddSource(cNewDens, cSDens);
	__syncthreads();

	cSwap(&cOldDens, &cNewDens);
	__syncthreads();
}

__global__ void cCalcDens(float* cNewDens, float* cOldDens) {
	cDiffuse(1, cNewDens, cOldDens);
	__syncthreads();
}

__global__ void cCalcAdv(float* cSVelX, float* cSVelY, float* cNewDens, float* cOldDens) {
	cSwap(&cOldDens, &cNewDens);
	__syncthreads();
}

__global__ void cCalcBound(float* cNewDens) {
	cSetBound(1, cNewDens);
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
__device__ void cDiffuse(int b, float* cNewDens, float* cOldDens) {
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

	//id is calculated with a -2*blockID to ensure that with a boundary to each block, every pixel onscreen will be calculated
	int ID = (blockIdx.x * blockDim.x + threadIdx.x - 2 * blockIdx.x) + (RES + 2)*(blockIdx.y * blockDim.y + threadIdx.y - 2 * blockIdx.y);

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

__device__ void cFinalProjection(float* cNewVelX, float* cNewVelY, float* cOldVelX) {
	int collumn = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = row * (RES + 2) + collumn;

	float h = 1.0f / RES;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cNewVelX[ID] -= 0.5f*(cOldVelX[ID + 1] - cOldVelX[ID - 1]) / h;
		cNewVelY[ID] -= 0.5f*(cOldVelX[ID + RES + 2] - cOldVelX[ID - RES - 2]) / h;

		cSetBound(1, cNewVelX);
		cSetBound(2, cNewVelY);
	}
}
__device__ void cProjectionInY(float* cNewVelX, float* cNewVelY, float* cOldVelX, float* cOldVelY) {
	int collumn = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = row * (RES + 2) + collumn;

	float h = 1.0f / RES;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cOldVelY[ID] = -0.5f * h * (cNewVelX[ID + 1] - cNewVelX[ID - 1]
			+ cNewVelY[ID + RES + 2] - cNewVelY[ID - RES - 2]);
		cOldVelX[ID] = 0;

		cSetBound(0, cOldVelY);
		cSetBound(0, cOldVelX);
	}
}
__device__ void cProjectionInX(float* cOldVelX, float* cOldVelY) {
	int collumn = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = row * (RES + 2) + collumn;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cOldVelX[ID] = (cOldVelY[ID] + cOldVelX[ID - 1]
			+ cOldVelX[ID + 1] + cOldVelX[ID - RES - 2] + cOldVelX[ID + RES + 2]) / 4;

		cSetBound(0, cOldVelX);
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
