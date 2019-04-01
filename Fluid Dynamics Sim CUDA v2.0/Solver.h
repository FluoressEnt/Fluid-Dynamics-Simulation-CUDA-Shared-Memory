#include "defines.h"


class Solver {
public:
	Solver();
	~Solver(void);

	void CalculateWrapper();

	float* GetDensityArray();
	float* GetVelXArray();
	float* GetVelYArray();

	void RefreshDensIn();
	void RefreshVelIn();
	void RefreshAll();

	void SetInputDens(int);
	void SetInputVel(int, int, int);

	float* GetInputDens();

private:

	//input arrays
	float *sDens;
	float* sVelX;
	float* sVelY;
	//CPU arrays
	float* newDens;
	float* oldDens;
	float* newVelX;
	float* newVelY;
	float* oldVelX;
	float* oldVelY;

	//pointers to arrays on gpu
	float* cSDens;
	float* cSVelX;
	float* cSVelY;
	float* cNewDens;
	float* cOldDens;
	float* cNewVelX;
	float* cNewVelY;
	float* cOldVelX;
	float* cOldVelY;
};

#ifdef __cplusplus
extern "C" {
#endif
	//kernels
	__global__ void cAddSource(float*, float*, float*);
	__global__ void cCalcDiffusion(float*, float*);
	__global__ void cCalcAdvection(float*, float*, float*, float*);
	__global__ void cCalcProjY(float*, float*, float*, float*);
	__global__ void cCalcProjX(float*, float*);
	__global__ void cCalcFinalProj(float*, float*, float*);
	__global__ void cCalcBound(int, float*);
	__global__ void cSwapPtr(float*, float*);

	//methods
	__device__ void cSwap(float**, float**);

#ifdef __cplusplus
}
#endif