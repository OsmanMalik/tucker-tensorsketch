/*
// 	SPTTTINNERPRODUCTC.C 	Computes the Frobenious inner product of a sparse 
//  				    	tensor and a Tucker tensor.
//
// INPUTS:
//	Xsubs 	- The subindices of the nonzero values of the sparse tensor. Should be a matrix formatted
// 			  as (tensor dim) x (nnz). Should be in int32 format. If, for example, X is an sptensor,
// 			  then the input would be int32(X.subs.').
// 	Xvals 	- A vector containing the nonzero values. This would be X.vals.
// 	G 		- This is the core tensor stored as a dense MATLAB double array.
// 	A 		- Row or column cell containing the transpose of the factor matrices. The factor matrices
// 			  are transposed since we want to access rows of each factor matrix at a time, and MATLAB
// 			  matrices are stored as 1-dim arrays in column major ordering.
//
// OUTPUTS:
// 	inner_prod 	- The output is the Frobenious inner product of the input sparse matrix and Tucker
// 				  matrix.
*/

/*
// Author:   Osman Asif Malik
// Email:    osman.malik@colorado.edu
// Date:     May 24, 2018
*/

#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Declare global variables for input sparse tensor and Tucker tensor */
int32_T *Xsubs;
double *Xvals;
mwSize Xnnz;
double *G;
double **A;
size_t no_A;
mwSize *R; 
mwSize *I;

/* Declare global variables for output scalar */
double *output_val;

/* Declare other global variables */
mwIndex *cum_dim_prod;
int32_T *current_X_idx;

/* Declare compute_ttensor_element function
   It computes a single element from the Tucker tensor defined by G and A using recursion */
double compute_ttensor_element(mwIndex prev_core_idx, size_t dim) {
	mwIndex r;
	mwIndex core_idx;
	double sum;
	
	sum = 0.0;
	for(r = 0; r < R[dim]; ++r) {
		core_idx = prev_core_idx + r*cum_dim_prod[dim];
		if(dim < no_A - 1) {
			sum += A[dim][r + (current_X_idx[dim] - 1)*R[dim]]
				* compute_ttensor_element(core_idx, dim + 1);
		} else {
			sum += A[dim][r + (current_X_idx[dim] - 1)*R[dim]] * G[core_idx];
		}
	}
	
	return sum;
}

/* Declare compute_norm function
   It computes the norm of the sparse matrix defined by Xsubs and Xvals, and the Tucker tensor
   defined by G and A */
void compute_norm() {
	mwIndex i; 
	double ip;
	
	ip = 0.0;
	current_X_idx = Xsubs;
	for(i = 0; i < Xnnz; ++i) {
		ip += Xvals[i]*compute_ttensor_element(0, 0);
		current_X_idx += no_A;
	}
	*output_val = ip;
}

/* mex interface */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Declare other variables */
	mwIndex i;
	
	/* Get input variables for sparse tensor and Tucker tensor */
	Xsubs = (int32_T *) mxGetData(prhs[0]);
	Xvals = mxGetPr(prhs[1]);
	Xnnz = mxGetDimensions(prhs[1])[0];
	G = mxGetPr(prhs[2]);
	no_A = mxGetNumberOfElements(prhs[3]);
	A = malloc(no_A*sizeof(double *));
	R = malloc(no_A*sizeof(mwSize));
	I = malloc(no_A*sizeof(mwSize));
	for(i = 0; i < no_A; ++i) {
		mxArray *current_A = mxGetCell(prhs[3], i);
		A[i] = mxGetPr(current_A);
		R[i] = mxGetDimensions(current_A)[0];
		I[i] = mxGetDimensions(current_A)[1];
	}
	
	/* Create the output scalar */
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	output_val = mxGetPr(plhs[0]);
	
	/* Compute output */
	cum_dim_prod = malloc(no_A*sizeof(mwIndex));
	cum_dim_prod[0] = 1;
	for(i = 1; i < no_A; ++i) {
		cum_dim_prod[i] = cum_dim_prod[i-1] * R[i-1];
	}
	
	compute_norm();
	
	/* Free dynamically allocated memory */
	free(cum_dim_prod);
	free(I);
	free(R);
	free(A);
}