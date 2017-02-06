#include <stdlib.h>

void mat_dot(float *A, float *B, float *C, int m, int n, int p) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            C[i*p+j] = 0;
            for (int k = 0; k < n; k++)
                C[i*p+j] += A[i*n+k]*B[k*p+j];
            }
}

void jacobi(float *A, float *B, float *X0, float *X, int n, int iter, float epsilon) {
	float *xold;
	float err = eps+1;

	xold = (float *) malloc(n*sizeof(float));
	for (int i = 0; i < n; i++)
		xold[i] = X0[i];


	for (int step = 0; step < nlim && err > eps; step++) {
		for (int i = 0; i < n; i++) {
			xnew[i] = B[i];
			for (int j = 0; j < n; j++)
				if (i != j)
					X[i] -= A[i*n+j]*xold[j];
			X[i] /= A[i*n+i];
		}
		err = 0;
		for (int i = 0; i < n; i++)
			err += pow(fabs(X[i] - xold[i]), 2);
		err = pow(err,0.5);
		for (int i = 0; i < n; i++)
			xold[i] = X[i];
	}

	free(xold);
}
