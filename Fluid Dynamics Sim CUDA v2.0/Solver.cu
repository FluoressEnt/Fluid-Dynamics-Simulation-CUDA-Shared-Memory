#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "Solver.h"

void Solver::CalculateWrapper() {
	//Shared memory block size declaration
	dim3 sBlockSize(BRES + 2, BRES + 2);
	//global memory block and grid declarations
	dim3 gBlockSize(BRES, BRES);
	dim3 gGridSize = dim3((RES + 2 + gBlockSize.x - 1) / gBlockSize.x, (RES + 2 + gBlockSize.y - 1) / gBlockSize.y);
	//dim3 gGridSize = dim3((RES + gBlockSize.x - 1) / gBlockSize.x, (RES + gBlockSize.y - 1) / gBlockSize.y);

	//diffusion part
	cAddSource << <gGridSize, gBlockSize >> > (cSDens, cNewDens, cOldDens);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcDiffusion << <gGridSize, sBlockSize>> > (cNewDens, cOldDens);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (1, cNewDens);
	}

	cSwapPtr << <gGridSize, gBlockSize >> > (cOldDens, cNewDens);
	cCalcAdvection << <gGridSize, sBlockSize >> > (cNewDens, cOldDens, cNewVelX, cNewVelY);
	cudaDeviceSynchronize();
	cCalcBound << <gGridSize, gBlockSize >> > (2, cNewDens);

	//velocity part
	cAddSource << <gGridSize, gBlockSize >> > (cSVelX, cNewVelX, cOldVelX);
	cAddSource << <gGridSize, gBlockSize >> > (cSVelY, cNewVelY, cOldVelY);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcDiffusion << <gGridSize, sBlockSize >> > (cNewVelX, cOldVelX);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
	}
	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcDiffusion << <gGridSize, sBlockSize>> > (cNewVelY, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);
	}

	cCalcProjY << <gGridSize, sBlockSize>> > (cNewVelX, cNewVelY, cOldVelX, cOldVelY);
	cudaDeviceSynchronize();
	cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelY);
	cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcProjX << <gGridSize, sBlockSize>> > (cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);
	}

	cCalcFinalProj << <gGridSize, sBlockSize >> > (cNewVelX, cNewVelY, cOldVelX);
	cudaDeviceSynchronize();
	cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
	cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);

	cSwapPtr << <gGridSize, gBlockSize >> > (cOldVelX, cNewVelX);
	cSwapPtr << <gGridSize, gBlockSize >> > (cOldVelY, cNewVelY);

	cCalcAdvection << <gGridSize, sBlockSize>> > (cNewVelX, cOldVelX, cOldVelX, cOldVelY);
	cudaDeviceSynchronize();
	cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
	cCalcAdvection << <gGridSize, sBlockSize>> > (cNewVelY, cOldVelY, cOldVelX, cOldVelY);
	cudaDeviceSynchronize();
	cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);

	cCalcProjY << <gGridSize, sBlockSize >> > (cNewVelX, cNewVelY, cOldVelX, cOldVelY);
	cudaDeviceSynchronize();
	cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelY);
	cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);

	for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
		cCalcProjX << <gGridSize, sBlockSize >> > (cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);
	}

	cCalcFinalProj << <gGridSize, sBlockSize >> > (cNewVelX, cNewVelY, cOldVelX);
	cudaDeviceSynchronize();
	cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
	cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);
}

//global memory kernel
__global__ void cAddSource(float* inputArr, float* arrayAffected, float* arrayToSwap) {
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = xPos + (RES + 2)*yPos;

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		float temp;
		temp = (DT * inputArr[ID]);
		arrayAffected[ID] += temp;
	}

	cSwap(&arrayToSwap, &arrayAffected);
}
//shared memory kernel
__global__ void cCalcDiffusion(float* newArray, float* oldArray) {
	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x -1;
	int yPos = blockIdx.y * BRES + threadIdx.y -1;
	int ID = xPos + (RES + 2)*yPos;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sNewDens[BRES + 2][BRES + 2];

	sNewDens[yDir][xDir] = newArray[ID];

	__syncthreads();

	//ensures a boundary is maintained around the block where only reads occur
	if ((yDir > 0) && (yDir <= BRES) && (xDir > 0) && (xDir <= BRES)) {

		newArray[ID] = (oldArray[ID] + A * (sNewDens[yDir][xDir - 1] + sNewDens[yDir][xDir + 1]
			+ sNewDens[yDir - 1][xDir] + sNewDens[yDir + 1][xDir])) / (1 + 4 * A);
	}
}
//shared memory kernel
__global__ void cCalcAdvection(float* cVelX, float* cVelY, float* cNew, float* cOld) {
	int left, bottom, right, top;
	float x, y, distToRight, distToTop, distToLeft, distToBottom;

	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x - 1;
	int yPos = blockIdx.y * BRES + threadIdx.y - 1;
	int ID = xPos + (RES + 2)*yPos;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sOldDens[BRES + 2][BRES + 2];
	sOldDens[yDir][xDir] = cOld[ID];
	__syncthreads();

	if ((yDir > 0) && (yDir <= BRES) && (xDir > 0) && (xDir <= BRES)) {

		x = xDir - DT * RES * cVelX[ID];
		y = yDir - DT * RES * cVelY[ID];

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

		cNew[ID] = distToRight * (distToTop * sOldDens[bottom][left] + distToBottom * sOldDens[top][left])
			+ distToLeft * (distToTop * sOldDens[bottom][right] + distToBottom * sOldDens[top][right]);
	}
}
//shared memory kernel
__global__ void cCalcProjY(float* cNewVelX, float* cNewVelY, float* cOldVelX, float* cOldVelY) {
	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x - 1;
	int yPos = blockIdx.y * BRES + threadIdx.y - 1;
	int ID = xPos + (RES + 2)*yPos;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sNewVelX[BRES + 2][BRES + 2];
	__shared__ float sNewVelY[BRES + 2][BRES + 2];

	sNewVelX[yDir][xDir] = cNewVelX[ID];
	sNewVelY[yDir][xDir] = cNewVelY[ID];
	__syncthreads();

	float h = 1.0f / RES;

	if ((yDir > 0) && (yDir <= BRES) && (xDir > 0) && (xDir <= BRES)) {
		cOldVelY[ID] = -0.5f * h * (sNewVelX[yDir][xDir + 1] - sNewVelX[yDir][xDir - 1]
			+ sNewVelY[yDir + 1][xDir] - sNewVelY[yDir - 1][xDir]);
		cOldVelX[ID] = 0;
	}
}
//shared memory kernel
__global__ void cCalcProjX(float* cOldVelX, float* cOldVelY) {
	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x - 1;
	int yPos = blockIdx.y * BRES + threadIdx.y - 1;
	int ID = xPos + (RES + 2)*yPos;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sOldVelX[BRES + 2][BRES + 2];

	sOldVelX[yDir][xDir] = cOldVelX[ID];
	__syncthreads();

	if ((yDir > 0) && (yDir <= BRES) && (xDir > 0) && (xDir <= BRES)) {
		cOldVelX[ID] = (cOldVelY[ID] + sOldVelX[yDir][xDir - 1]
			+ sOldVelX[yDir][xDir + 1] + sOldVelX[yDir - 1][xDir] + sOldVelX[yDir + 1][xDir]) / 4;
	}
}
//shared memory kernel
__global__ void cCalcFinalProj(float* cNewVelX, float* cNewVelY, float* cOldVelX) {
	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x - 1;
	int yPos = blockIdx.y * BRES + threadIdx.y - 1;
	int ID = xPos + (RES + 2)*yPos;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sOldVelX[BRES + 2][BRES + 2];
	sOldVelX[yDir][xDir] = cOldVelX[ID];
	__syncthreads();

	float h = 1.0f / RES;

	if ((yDir > 0) && (yDir <= BRES) && (xDir > 0) && (xDir <= BRES)) {
		cNewVelX[ID] -= 0.5f*(sOldVelX[yDir][xDir + 1] - sOldVelX[yDir][xDir - 1]) / h;
		cNewVelY[ID] -= 0.5f*(sOldVelX[yDir + 1][xDir] - sOldVelX[yDir - 1][xDir]) / h;
	}
}

//global memory kernel
__global__ void cCalcBound(int b, float* boundArray) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int collumn = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = row + (RES + 2) * collumn;

	int xPos, yPos;
	xPos = ID % (RES + 2);
	yPos = ID / (RES + 2);

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

//Pointer swap kernel & method
__global__ void cSwapPtr(float* arrayOne, float* arrayTwo) {
	cSwap(&arrayOne, &arrayTwo);
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