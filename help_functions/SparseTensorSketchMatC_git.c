/*
// SPARSETENSORSKETCHMATC.C 	Computes the sketch of the transpose of mode-n matricization
// 								of the input tensor. The sketch is the TensorSketch 
// 								corresponding to the input hashing functions h and s.
//
// INPUTS:
//	Y.vals		- This is a row vector containing the non-zero values of the tensor. This
// 				  should be a double array.
// 	Y.subs 		- This is a int32 matrix containing the subindicies for the non-zero entries
// 				  of the input tensor. The matrix should be of size (no_dim) x (nnz). The
//				  reason for this is to ensure efficient memory access when computing the
// 				  sketch.
// 	h 			- A row or column cell containing column vectors representing the hashing 
// 				  functions. Each column vector should be of type int32.
// 	s 			- A row or column cell containing column vectors representing the sign
// 				  hashing function. The column vectors should be left as double arrays.
// 	sketch_dim	- The target sketch dimension. Should be of type int32.
// 	no_cols 	- The number of columns of the sketched matrix, i.e., no_cols = size(Y,n).
// 	n 			- The mode along which Y is matricized.
//
// OUTPUTS:
// 	MsT 		- A TensorSketched version of the transpose of the mode-n matricization of
// 				  the input tensor. MsT will have sketch_dim rows and no_cols columns.
//
// NOTES:
// 	Note 1 		- This C function is meant to handle tensors of the sptensor type from the
// 				  TensorToolbox. The reason things are passed in one-by-one as separate
// 				  arguments (Y.vals, Y.subs, no_cols) instead of as a sptensor structure
// 				  is that Y.subs is stored as a a (nnz) x (no_dim) matrix in sptensor.subs.
// 				  By using the transpose of that matrix instead, memory access is more 
// 				  efficient.
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

/* Declare sparse input matrix global variables */
double *input_values;
int64_T *input_indices; /* Should be inserted as a no_dim x nnz array */
mwSize no_dim;
mwSize nnz;

/* Declare hashing and sign function global variables */
int64_T **h; /* Will point to array of int64_T * pointing to each hashing function */
double **s; /* Will point to array of double * pointing to each sign function */
size_t no_hash_func; /* Will store number of hashing functions */
size_t *len; /* Will point to array of length of each hashing/sign function */

/* Declare output matrix global variables */
double *output_matrix;
int64_T sketch_dim; /* Number of rows in sketched matrix */
int64_T no_cols;

/* Declare other global variables */
int64_T n;

/* Declare and define functions */
void compute_sketch() {
	mwIndex i, j, idx, sum, target_row;
	double prod;
	for(i = 0; i < nnz; ++i) {
		sum = 0;
		prod = 1.0;
		for(j = 0; j < no_hash_func; ++j) {
			if(j < n - 1) {
				idx = j;
			} else {
				idx = j + 1;
			}
			sum += h[j][input_indices[idx + i*no_dim] - 1];
			prod *= s[j][input_indices[idx + i*no_dim] - 1];
		}
		target_row = (sum - no_hash_func) % sketch_dim;
		output_matrix[target_row + sketch_dim*(input_indices[n-1 + i*no_dim] - 1)] += prod*input_values[i];
	}
}

/* mex interface */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Declare other variables */
	mwIndex i;
	
	/* Get sparse input matrix with its dimensions */
	input_values = mxGetPr(prhs[0]);
	input_indices = (int64_T *) mxGetData(prhs[1]);
	no_dim = mxGetDimensions(prhs[1])[0];
	nnz = mxGetDimensions(prhs[1])[1];
	
	/* Get hashing/sign functions with their dimensions */
	no_hash_func = mxGetNumberOfElements(prhs[2]);
	h = malloc(no_hash_func*sizeof(int64_T *));
	s = malloc(no_hash_func*sizeof(double *));
	len = malloc(no_hash_func*sizeof(size_t));
	for(i = 0; i < no_hash_func; ++i) {
		h[i] = (int64_T *) mxGetData(mxGetCell(prhs[2], i));
		s[i] = mxGetPr(mxGetCell(prhs[3], i));
		len[i] = mxGetM(mxGetCell(prhs[2], i));
	}

	/* Create the output matrix */
	sketch_dim = *((int64_T *) mxGetData(prhs[4]));
	no_cols = *((int64_T *) mxGetData(prhs[5]));
	plhs[0] = mxCreateDoubleMatrix(sketch_dim, no_cols, mxREAL);
	output_matrix = mxGetPr(plhs[0]);
	n = *((int64_T *) mxGetData(prhs[6]));
	
	/* Compute output matrix */
	compute_sketch();
	
	/* Free dynamically allocated memory */
	free(len);
	free(s);
	free(h);	
}