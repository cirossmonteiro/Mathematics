#include <cuda_runtime.h>
#include <math.h>

__global__ void CUDA_mat_dot_kernel(float *A, float *B, float *C, int m, int n, int p) {
    int ind = blockIdx.x*blockDim.x+threadIdx.x;
    int i = ind/p, j = ind%p;
    if (ind >= m*p)
        return;
    for (int k = 0; k < n; k++)
        C[ind] += A[i*n+k]*B[k*p+j];
}

void CUDA_mat_dot(float *A, float *B, float *C, int m, int n, int p) {
    float *dA, *dB, *dC;

    int thpb; // max: 1024
    int blpg; // max: 65536
    int num = m*p;

    if (num <= 65536*32) {
		thpb = 32;
		blpg = num/32+1;
		if (num == 65536*32)
			blpg--;
	}
	else {
		thpb = num/65536+1;
		blpg = 65536;
    }

    cudaMalloc(&dA, m*n*sizeof(float));
    cudaMalloc(&dB, n*p*sizeof(float));
    cudaMalloc(&dC, m*p*sizeof(float));
    cudaMemcpy(dA, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n*p*sizeof(float), cudaMemcpyHostToDevice);

    CUDA_mat_dot_kernel<<<blpg,thpb>>>(dA, dB, dC, m, n, p);
    cudaDeviceSynchronize(); // not sure if this makes any difference

    cudaMemcpy(C, dC, m*p*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

__global__ void CUDA_jacobi_kernel(double *A, double *B, double *xold, double *xnew, int n) {
	int ind = blockIdx.x*blockDim.x+threadIdx.x;
	double s;
	if (ind >= n)
		return;
	s = B[ind];
	for (int j = 0; j < n; j++)
		if (ind != j)
			s -= A[ind*n+j]*xold[j];
	xnew[ind] = s / A[ind*n+ind];
}

void CUDA_jacobi(float *A, float *B, float *X0, float *X, int n, int iter, float epsilon) {
	int thpb; // max: 1024
	int blpg; // max: 65536
	float *dXold, *dXnew, *dA, *dB, *hXold;
	float err;

	if (n <= 65536*32) {
		thpb = 32;
		blpg = n/32+1;
		if (num == 65536*32)
			blpg--;
	}
	else {
		thpb = num/65536+1;
		blpg = 65536;
	}

	hXold = (float *) calloc(num, sizeof(float));

	cudaMalloc(&dXold, n*sizeof(float));
	cudaMalloc(&dXnew, n*sizeof(float));
	cudaMalloc(&dA, n*n*sizeof(float));
	cudaMalloc(&dB, n*sizeof(float));

	cudaMemcpy(dA, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, n*sizeof(float), cudaMemcpyHostToDevice);

	// first iteration
	cudaMemcpy(dXold, X0, n*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_jacobi_kernel<<<blpg,thpb>>>(dA, dB, dXold, dXnew, n); // do here
	cudaMemcpy(X, dXnew, n*sizeof(float), cudaMemcpyDeviceToHost);

	// compute error
	err = 0;
	for (int i = 0; i < num; i++)
		err += pow(fabs(X[i] - X[i]),2);
	err = pow(err,0.5);

	// other iterations
	for (int i = 0; i < nlim-1 && err > eps; i++) {
		cudaMemcpy(dXold, dXnew, n*sizeof(float), cudaMemcpyDeviceToDevice);
		CUDA_jacobi_kernel<<<blpg,thpb>>>(dA, dB, dXold, dXnew, num);
		cudaMemcpy(hXold, dXold, n*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(X, dXnew, n*sizeof(float), cudaMemcpyDeviceToHost);

		// compute error
		err = 0;
		for (int j = 0; j < num; j++)
			err += pow(fabs(X[j] - hXold[j]), 2);
		err = pow(err, 0.5);
	}

	cudaFree(dXold);
	cudaFree(dXnew);
	cudaFree(dA);
	cudaFree(dB);
	free(hXold);
}
