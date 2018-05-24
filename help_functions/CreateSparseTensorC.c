/*
//	CREATESPARSETENSORC.C 	Computes a sparse tensor based on a dense core tensor and sparse factor
// 							matrices. The outputs are a matrix containing the indices of the nonzero
// 							elements of the output sparse tensor, and a vector containing the corres-
// 							ponding nonzero values.
//
// INPUTS:
// 	Asubs 				- This is a (row or column) cell, with the n-th cell entry containing 
// 						  the indices of the
// 						  nonzero elements in the n-th factor matrix. For each factor matrix A,
// 						  these indices are stored as a 2 x nnz(A) matrix. The reason the indices
// 						  are stored like this instead of a nnz(A) x 2 matrix (which would follow
// 						  the convention of the Tensor Toolbox) is that keeping it as a short and
// 						  fat matrix makes the memory access more efficient in this code. Note that
// 						  all entries must be of type int32. IMPORTANT: The subindices must be sorted
// 						  according to row index; this can be achieved by applying sortrows to the matrix
// 						  before transposing.
// 	Avals 				- This is a (row of column) cell, with the n-th cell entry containing a 
// 						  MATLAB double vector containing the nonzero entries of the n-th factor
// 						  matrix, ordered in the same order as the corresponding indices in Asubs.
// 					      IMPORTANT: Avals must be sorted in the same way as Asubs.
//	G 					- This is the core tensor in the form of a dense MATLAB double array.
// 	R 					- This is an int32 MATLAB vector which contains the multi-rank of the tensor.
// 	no_output_vals 		- This is an int32 value which gives the number of nonzero elements in the 
// 						  output tensor. The reason this is provided as a parameter rather than computed
// 						  inside this function is that it can be computed very efficiently in MATLAB,
// 						  which avoids making this C code more complicated than necessary.
//
// OUTPUTS:
// 	Tsubs 				- This is a dim x nnz int32 matrix which contains the indices of the nonzero
// 						  entries in the output tensor, where dim in the dimension of the output tensor.
// 						  Again, the reason this matrix is short and fat rather than the other way around
// 						  is to optimize memory access. Before inserting this in Tensor Toolbox to create
// 					 	  an sptensor, Tsubs needs to be transposed and converted to double in MATLAB.
// 	Tvals 				- This is a double vector which contains the nonzero values corresponding to the
// 						  indices in Tsubs.
//
// NOTES:
// 	Note 1 	- To see how to use this function, and how to convert the output to a Tensor Toolbox sptensor,
// 			  see the code in mextest5.
// 	Note 2 	- The general idea of the algorithm implemented in this code is the following. The output tensor
// 			  will contain a certain number of nonzero elements. Each such nonzero element is a sum of products,
// 			  where each such product is composed of one element from each factor matrix and one element from the
// 			  core tensor. In order to avoid having to identify all products than contribute to a certain tensor
// 			  entry, or going through products one by one and adding them to the appropriate output tensor 
// 			  element, we can instead first sort the input Asubs and Avals, and then arrange the computations
// 			  in this code so that one tensor element is computed fully before proceeding to the next one. 
//  		  This is acomplished by using two recursive functions: compute_output() will recursively go
// 			  through the factor matrices and identify "blocks" in each matrix, where a block is a set of
// 			  indices with the same row index. For each configuration of blocks from each factor matrix,
// 			  the compute_output() will call compute_tensor_value(), which computes the sum of all products of
// 			  the factor matrix elements in the given blocks (and appropriate core tensor element) in an
// 			  iterative fashion. Since this sum is for elements coming from the same row of their respective
// 			  factor matrices, it means that they all belong on in the same entry in the output tensor.
// 	Note 3 	- In order to avoid excessive passing of parameters between functions, quite a few variables
// 			  have been made global.
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

/* Declare global variables for factor matrices and core tensor */
size_t no_A;
int32_T **Asubs;
double **Avals;
mwSize *no_vals_A;
double *G;
int32_T *R;

/* Declare global variables for output sparse tensor */
int32_T no_output_vals;
int32_T *output_subs;
double *output_vals;

/* Declare other global variables */
mwIndex output_sub_idx;
mwIndex output_val_idx;
mwIndex *cum_dim_prod;
mwIndex *block_start;

/* Declare compute_tensor_value function
   Computes the value of a tensor for a specific index */
void compute_tensor_value(double prev_prod, mwIndex prev_core_idx, size_t dim) {
	mwIndex idx, i, core_idx;
	double prod;
	i = block_start[dim];
	idx = Asubs[dim][2*i];
	while(Asubs[dim][2*i] == idx && i < no_vals_A[dim]) {
		prod = Avals[dim][i] * prev_prod;
		core_idx = (Asubs[dim][2*i+1] - 1)*cum_dim_prod[dim] + prev_core_idx;
		if(dim < no_A - 1) {
			compute_tensor_value(prod, core_idx, dim + 1);
		} else {
			output_vals[output_val_idx] += G[core_idx] * prod;
		}
		++i;
	}
}

/* Declare compute_output function
   Goes through all "block" corresponding to the same tensor index */
void compute_output(size_t dim) {
	mwIndex j, current_idx;
	int break_flag;
	break_flag = 0;
	while(true) {		
		/* Compute tensor value, or go one level deeper */
		if(dim < no_A - 1) {
			compute_output(dim + 1);
		} else {
			mwIndex i;
			compute_tensor_value(1.0, 0, 0);
			++output_val_idx;
			for(i = 0; i < no_A; ++i) {
				output_subs[output_sub_idx++] = Asubs[i][2*block_start[i]];
			}
		}
		
		/* Determine next block start */
		current_idx = Asubs[dim][2*block_start[dim]];
		j = block_start[dim] + 1;
		break_flag = 1;
		while(j < no_vals_A[dim]) {
			if(Asubs[dim][2*j] != current_idx) {
				block_start[dim] = j;
				break_flag = 0;
				break;
			}
			++j;
		}
		
		/* Break if there are no more blocks */
		if(break_flag == 1) {
			block_start[dim] = 0;
			break;
		}
	}
}

/* mex interface */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Declare other variables */
	mwIndex i;
	
	/* Get input variables for factor matrices and core tensor */
	no_A = mxGetNumberOfElements(prhs[0]);
	Asubs = malloc(no_A*sizeof(int32_T *));
	Avals = malloc(no_A*sizeof(double *));
	no_vals_A = malloc(no_A*sizeof(mwSize));
	for(i = 0; i < no_A; ++i) {
		mxArray *current_A = mxGetCell(prhs[0], i);
		Asubs[i] = (int32_T *) mxGetData(current_A);
		Avals[i] = mxGetPr(mxGetCell(prhs[1], i));
		no_vals_A[i] = mxGetDimensions(current_A)[1];
	}
	
	G = mxGetPr(prhs[2]);
	R = (int32_T *) mxGetData(prhs[3]);
		
	/* Create the output matrix subs and vals */
	no_output_vals = *((int32_T *) mxGetData(prhs[4]));
	plhs[0] = mxCreateNumericMatrix(no_A, no_output_vals, mxINT32_CLASS, mxREAL);
	output_subs = (int32_T *) mxGetData(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(no_output_vals, 1, mxREAL);
	output_vals = mxGetPr(plhs[1]);	
		
	/* Compute output */
	output_sub_idx = 0;
	output_val_idx = 0;
	cum_dim_prod = malloc(no_A*sizeof(mwIndex));
	cum_dim_prod[0] = 1;
	for(i = 1; i < no_A; ++i) {
		cum_dim_prod[i] = cum_dim_prod[i-1] * R[i-1];
	}
	block_start = (mwIndex *) calloc(no_A, sizeof(mwIndex));
	compute_output(0);
	
	/* Free dynamically allocated memory */
	free(block_start);
	free(cum_dim_prod);
	free(no_vals_A);
	free(Avals);
	free(Asubs);	
}