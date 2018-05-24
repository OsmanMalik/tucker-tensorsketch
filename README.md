# tucker-tensorsketch
Matlab function for low-rank Tucker decomposition of tensors using TensorSketch.

## Compiling mex files
Run the file **compile_all_mex.m** inside the folder help_functions. Alternatively, simply compile each c file individually by running "mex filename.c" inside Matlab.

## Requirements
This code requires Tensor Toolbox version 2.6 by Bader, Kolda and others (available at http://www.sandia.gov/~tgkolda/TensorToolbox/).

## Demo files
The three demo files demonstrate Tucker decomposition of sparse tensors, dense tensors, and dense tensors stored on disk in mat files.
