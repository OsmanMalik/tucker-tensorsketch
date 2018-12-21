/*
// SPARSETENSORSKETCHMATDSC.C 	Computes the sketch of the transpose of mode-n matricization
// 								of the input tensor, as well as the two corresponding double-
//							    sketches. It returns all three parameters. The first sketch is 
//								the TensorSketch corresponding to the input hash functions h
//								and sign functions s. The double-sketches are CountSketches
//								corresponding to h_n1 and s_n, as well as h_n2 and s_n.
//
// INPUTS:
//	[0]  Y.vals			- This is a row vector containing the non-zero values of the tensor. This
// 				  	  	  should be a double array.
// 	[1]  Y.subs 		- This is a int32 matrix containing the subindicies for the non-zero entries
// 				  	  	  of the input tensor. The matrix should be of size (no_dim) x (nnz). The
//				  	  	  reason for this is to ensure efficient memory access when computing the
// 				  	  	  sketch.
// 	[2]  h 				- A row or column cell containing column vectors representing the hashing 
// 				  	  	  functions. Each column vector should be of type int32.
//	[3]  h_n1			- A column vector corresponding to the n-th hash function which we use to 
//				  	  	  CountSketch for dimension s1 with. Should be int32.
//	[4]  h_n2			- A column vector corresponding to the n-th hash function which we use to
//				  	  	  CountSketch for dimension s2 with. Should be int32.
// 	[5]  s 				- A row or column cell containing column vectors representing the sign
// 				  	  	  hashing function. The column vectors should be left as double arrays.
//	[6]  s_n			- This is the sign function, in the form of a column vector, representing
//				  	  	  the n-th sign function which we use for the double CountSketches. Should
//				  	  	  be double.
// 	[7]  sketch_dim		- The smaller target sketch dimension. Should be of type int32.
//	[8]  sketch_dim2	- The larger target sketch dimension. Should be of type int32.
// 	[9]  no_cols 		- The number of columns of the sketched matrix, i.e., no_cols = size(Y,n).
// 	[10] n 				- The mode along which Y is matricized.
//
// OUTPUTS:
// 	[0] MsT 	- A TensorSketched version of the transpose of the mode-n matricization of
// 				  the input tensor. MsT will have sketch_dim rows and no_cols columns.
//	[1] MsTDS1 	- The first double-sketched matrix.
//	[2] MsTDS2 	- The second double-sketched matrix.
//
// NOTES:
// 	Note 1 		- This C function is meant to handle tensors of the sptensor type from the
// 				  TensorToolbox. The reason things are passed in one-by-one as separate
// 				  arguments (Y.vals, Y.subs, no_cols) instead of as a sptensor structure
// 				  is that Y.subs is stored as a a (nnz) x (no_dim) matrix in sptensor.subs.
// 				  By using the transpose of that matrix instead, memory access is more 
// 				  efficient.
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
int64_T *h_n1;
int64_T *h_n2;
double **s; /* Will point to array of double * pointing to each sign function */
double *s_n;
size_t no_hash_func; /* Will store number of hashing functions */
size_t *len; /* Will point to array of length of each hashing/sign function */

/* Declare output matrix global variables */
double *output_matrix;
double *output_matrix_DS1;
double *output_matrix_DS2;
int64_T sketch_dim; /* Number of rows in sketched matrix */
int64_T sketch_dim2;
int64_T no_cols;

/* Declare other global variables */
int64_T n;

/* Declare and define functions */
void compute_sketch() {
	mwIndex i, j, idx, sum, target_row, target_col;
	double prod, update_val;
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
		update_val = prod*input_values[i];
		target_row = (sum - no_hash_func) % sketch_dim;
		target_col = input_indices[n-1 + i*no_dim] - 1;
		output_matrix[target_row + sketch_dim*target_col] += update_val;
		output_matrix_DS1[target_row + sketch_dim*(h_n1[target_col] - 1)] += s_n[target_col]*update_val;
		output_matrix_DS2[target_row + sketch_dim*(h_n2[target_col] - 1)] += s_n[target_col]*update_val;
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
		s[i] = mxGetPr(mxGetCell(prhs[5], i));
		len[i] = mxGetM(mxGetCell(prhs[2], i));
	}
	h_n1 = (int64_T *) mxGetData(prhs[3]);
	h_n2 = (int64_T *) mxGetData(prhs[4]);
	s_n = mxGetPr(prhs[6]);

	/* Create the output matrices */
	sketch_dim = *((int64_T *) mxGetData(prhs[7]));
	sketch_dim2 = *((int64_T *) mxGetData(prhs[8]));
	no_cols = *((int64_T *) mxGetData(prhs[9]));
	plhs[0] = mxCreateDoubleMatrix(sketch_dim, no_cols, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(sketch_dim, sketch_dim, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(sketch_dim, sketch_dim2, mxREAL);
	output_matrix = mxGetPr(plhs[0]);
	output_matrix_DS1 = mxGetPr(plhs[1]);
	output_matrix_DS2 = mxGetPr(plhs[2]);
	n = *((int64_T *) mxGetData(prhs[10]));
	
	/* Compute output matrix */
	compute_sketch();
	
	/* Free dynamically allocated memory */
	free(len);
	free(s);
	free(h);	
}