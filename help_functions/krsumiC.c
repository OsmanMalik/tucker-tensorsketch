/* 
// KRSUMIC.C	Computes the Khatri-Rao product of the COMPLEX matrics in the input cell and returns the column vector 
//				we get by summing the elements in each row. 
//
// INPUTS:
//	A			- This function takes a single input in the form of a cell containing matrices. The matrices can be
//				  complex valued. All matrices must have the same number of columns.
//
// OUTPUTS:
//	krsum		- The function computes the sum of each row in the Khatri-Rao product of the matrices in A and returns 
//				  the resulting column vector.
*/

/*
// Author:   Osman Asif Malik
// Email:    osman.malik@colorado.edu
// Date:     September 17, 2018
*/

#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Declare input global variables */
double **A_r;
double **A_i;
size_t no_input_matrices;
mwIndex no_cols;
mwIndex *no_rows;

/* Declare output global variables */
double *output_matrix_r;
double *output_matrix_i;
mwIndex no_output_rows;

/* Declare other variables */
mwIndex idx; /* Used to keep track of next entry to compute in output matrix */

void compute_kr(double old_prod_r, double old_prod_i, mwIndex dim, mwIndex col) {
	mwIndex i;
	double prod_r, prod_i;
	
	for(i = 0; i < no_rows[dim]; ++i) {
		prod_r = old_prod_r*A_r[dim][i + col*no_rows[dim]] - old_prod_i*A_i[dim][i + col*no_rows[dim]];
		prod_i = old_prod_i*A_r[dim][i + col*no_rows[dim]] + old_prod_r*A_i[dim][i + col*no_rows[dim]];
		if(dim < no_input_matrices - 1) {
			compute_kr(prod_r, prod_i, dim + 1, col);
		} else {
			output_matrix_r[idx] += prod_r;
			output_matrix_i[idx] += prod_i;			
			++idx;
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Declare other variables */
	mwIndex i;
		
	/* Get input matrices */
	no_input_matrices = mxGetNumberOfElements(prhs[0]);
	A_r = malloc(no_input_matrices*sizeof(double *));
	A_i = malloc(no_input_matrices*sizeof(double *));
	no_rows = malloc(no_input_matrices*sizeof(mwIndex));
	no_cols = mxGetN(mxGetCell(prhs[0], 0)); /* We assume all input matrices has the same no cols */
	for(i = 0; i < no_input_matrices; ++i) {
		A_r[i] = mxGetPr(mxGetCell(prhs[0], i));
		A_i[i] = mxGetPi(mxGetCell(prhs[0], i));
		no_rows[i] = mxGetM(mxGetCell(prhs[0], i));
	}
	
	/* Create output matrix */
	no_output_rows = 1;
	for(i = 0; i < no_input_matrices; ++i) {
		no_output_rows *= no_rows[i];
	}
	plhs[0] = mxCreateDoubleMatrix(no_output_rows, 1, mxCOMPLEX);
	output_matrix_r = mxGetPr(plhs[0]);
	output_matrix_i = mxGetPi(plhs[0]);
	
	for(i = 0; i < no_cols; ++i) {
		idx = 0;
		compute_kr(1.0, 0.0, 0, i);
	}
	
	/* Free dynamically allocated memory */
	free(no_rows);
	free(A_i);
	free(A_r);
}