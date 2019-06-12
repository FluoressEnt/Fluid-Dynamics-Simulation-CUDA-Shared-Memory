#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "Solver.h"

///function to swap the pointers of two arrays
void swap(float **a, float **b) {
	float *t = *a;
	*a = *b;
	*b = t;
}

///Creates kernels that run in parallel on the GPU
void Solver::CalculateWrapper() {
	//Shared memory block size declaration
	dim3 sBlockSize(BRES + 2, BRES + 2);
	//global memory block and grid declarations
	dim3 gBlockSize(BRES, BRES);

	dim3 gGridSize = dim3((RES + 2 + gBlockSize.x - 1) / gBlockSize.x, (RES + 2 + gBlockSize.y - 1) / gBlockSize.y);

	//diffusion part

	// Add source 
	{
		cAddSource << <gGridSize, gBlockSize >> > (cSDens, cNewDens);
		cudaDeviceSynchronize();
	}

	swap(&cNewDens, &cOldDens);

	// Diffusion
	{
		for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
			cCalcDiffusion << <gGridSize, sBlockSize >> > (cNewDens, cOldDens);
			cudaDeviceSynchronize();
			cCalcBound << <gGridSize, gBlockSize >> > (1, cNewDens);
			cudaDeviceSynchronize();
		}
	}

	swap(&cNewDens, &cOldDens);

	// Advection
	{
		cCalcAdvection << <gGridSize, sBlockSize >> > (cNewDens, cOldDens, cNewVelX, cNewVelY);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (2, cNewDens);
		cudaDeviceSynchronize();
	}

	//velocity part

	// Add Source 
	{
		cAddSource << <gGridSize, gBlockSize >> > (cSVelX, cNewVelX);
		cAddSource << <gGridSize, gBlockSize >> > (cSVelY, cNewVelY);
		cudaDeviceSynchronize();
	}

	swap(&cNewVelX, &cOldVelX);
	swap(&cNewVelY, &cOldVelY);

	// Diffusion
	{
		for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
			cCalcDiffusion << <gGridSize, sBlockSize >> > (cNewVelX, cOldVelX);
			cCalcDiffusion << <gGridSize, sBlockSize >> > (cNewVelY, cOldVelY);
			cudaDeviceSynchronize();
			cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
			cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);
			cudaDeviceSynchronize();
		}
	}

	// Projection 
	{
		cCalcProjY << <gGridSize, sBlockSize >> > (cNewVelX, cNewVelY, cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelY);
		cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);
		cudaDeviceSynchronize();

		for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
			cCalcProjX << <gGridSize, sBlockSize >> > (cOldVelX, cOldVelY);
			cudaDeviceSynchronize();
			cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);
			cudaDeviceSynchronize();
		}

		cCalcFinalProj << <gGridSize, sBlockSize >> > (cNewVelX, cNewVelY, cOldVelX);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
		cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);
		cudaDeviceSynchronize();
	}

	swap(&cNewVelX, &cOldVelX);
	swap(&cNewVelY, &cOldVelY);

	// Advection 
	{
		cCalcAdvection << <gGridSize, sBlockSize >> > (cNewVelX, cOldVelX, cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcAdvection << <gGridSize, sBlockSize >> > (cNewVelY, cOldVelY, cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
		cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);
		cudaDeviceSynchronize();
	}

	// Projection
	{
		cCalcProjY << <gGridSize, sBlockSize >> > (cNewVelX, cNewVelY, cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelY);
		cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);
		cudaDeviceSynchronize();

		for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
			cCalcProjX << <gGridSize, sBlockSize >> > (cOldVelX, cOldVelY);
			cudaDeviceSynchronize();
			cCalcBound << <gGridSize, gBlockSize >> > (0, cOldVelX);
			cudaDeviceSynchronize();
		}

		cCalcFinalProj << <gGridSize, sBlockSize >> > (cNewVelX, cNewVelY, cOldVelX);
		cudaDeviceSynchronize();
		cCalcBound << <gGridSize, gBlockSize >> > (1, cNewVelX);
		cCalcBound << <gGridSize, gBlockSize >> > (2, cNewVelY);
		cudaDeviceSynchronize();
	}
}


///global memory kernel that adds a source to array values
__global__ void cAddSource(float* inputArr, float* arrayAffected) {
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;
	int ID = xPos + (RES + 2)*yPos;

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		float temp;
		temp = (DT * inputArr[ID]);
		arrayAffected[ID] += temp;
	}
}
///shared memory kernel that calculates the diffusion of the fluid on th GPU
__global__ void cCalcDiffusion(float* newArray, float* oldArray) {
	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x + 1;
	int yPos = blockIdx.y * BRES + threadIdx.y + 1;
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
///shared memory kernel that calculates the advection of the fluid on the GPU
__global__ void cCalcAdvection(float* cNew, float* cOld, float* cVelX, float* cVelY) {
	int left, bottom, right, top;
	float x, y, distToRight, distToTop, distToLeft, distToBottom;

	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x + 1;
	int yPos = blockIdx.y * BRES + threadIdx.y + 1;
	int ID = xPos + (RES + 2)*yPos;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sOldDens[BRES + 2][BRES + 2];
	sOldDens[yDir][xDir] = cOld[ID];
	__syncthreads();

	if ((yDir > 0) && (yDir <= BRES) && (xDir > 0) && (xDir <= BRES)) {

		x = xDir - DT * RES * cVelX[ID];
		y = yDir - DT * RES * cVelY[ID];

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
///shared memory kernel that calculates a part of the projection of the fluid on the GPU
__global__ void cCalcProjY(float* cNewVelX, float* cNewVelY, float* cOldVelX, float* cOldVelY) {
	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x + 1;
	int yPos = blockIdx.y * BRES + threadIdx.y + 1;
	int ID = xPos + (RES + 2)*yPos;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sNewVelX[BRES + 2][BRES + 2];
	__shared__ float sNewVelY[BRES + 2][BRES + 2];

	sNewVelX[yDir][xDir] = cNewVelX[ID];
	sNewVelY[yDir][xDir] = cNewVelY[ID];
	__syncthreads();

	float h = 1.0f / BRES;

	if ((yDir > 0) && (yDir <= BRES) && (xDir > 0) && (xDir <= BRES)) {
		cOldVelY[ID] = -0.5f * h * (sNewVelX[yDir][xDir + 1] - sNewVelX[yDir][xDir - 1]
			+ sNewVelY[yDir + 1][xDir] - sNewVelY[yDir - 1][xDir]);
		cOldVelX[ID] = 0;
	}
}
///shared memory kernel that calculates a part of the projection of the fluid on the GPU
__global__ void cCalcProjX(float* cOldVelX, float* cOldVelY) {
	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x + 1;
	int yPos = blockIdx.y * BRES + threadIdx.y + 1;
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
///shared memory kernel that calculates a part of the projection of the fluid on the GPU
__global__ void cCalcFinalProj(float* cNewVelX, float* cNewVelY, float* cOldVelX) {
	int xDir = threadIdx.x;
	int yDir = threadIdx.y;

	int xPos = blockIdx.x * BRES + threadIdx.x + 1;
	int yPos = blockIdx.y * BRES + threadIdx.y + 1;
	int ID = xPos + (RES + 2)*yPos;

	//creating shared memory and populating it
	//each thread fills an element of the block
	__shared__ float sOldVelX[BRES + 2][BRES + 2];
	sOldVelX[yDir][xDir] = cOldVelX[ID];
	__syncthreads();

	float h = 1.0f / BRES;

	if ((yDir > 0) && (yDir <= BRES) && (xDir > 0) && (xDir <= BRES)) {
		cNewVelX[ID] -= 0.5f*(sOldVelX[yDir][xDir + 1] - sOldVelX[yDir][xDir - 1]) / h;
		cNewVelY[ID] -= 0.5f*(sOldVelX[yDir + 1][xDir] - sOldVelX[yDir - 1][xDir]) / h;
	}
}

///global memory kernel that sets the boundary of the fluid to one where fluid does not leave the grid
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
	else if (xPos == 0 && yPos > 0 && yPos <= RES) {
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
