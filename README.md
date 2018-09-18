# Tucker-TensorSketch
Matlab functions for low-rank Tucker decomposition of tensors using TensorSketch.

## Some further details
Tucker-TensorSketch provides two functions, **tucker_ts** and **tucker_ttmts**, for low-rank Tucker decomposition of tensors. Both functions are variants of the standard alternating least-squares algorithm (higher-order orthogonal iteration) for the Tucker decomposition. They both incorporate a sketching technique called TensorSketch, which is a form of CountSketch that can be applied efficiently to matrices that are Kronecker products of smaller matrices. 

Due to the properties of TensorSketch, our functions only require a single pass of the input tensor. They can handle streamed data in the sense that they can read tensor elements in any order, and do not need to have access to all elements at the same time. The functions can handle larger tensors than competing algorithms, without sacrificing too much in terms of accuracy. 

![test](Experiment2Fig1.png)

## Requirements
This code requires Tensor Toolbox version 2.6 by Bader, Kolda and others (available at http://www.sandia.gov/~tgkolda/TensorToolbox/).

## Installation
Run the file **compile_all_mex.m** inside the folder help_functions. Alternatively, simply compile each c file individually by running "mex filename.c" inside Matlab.

## Demo files
The three demo files demonstrate Tucker decomposition of sparse tensors, dense tensors, and dense tensors stored on disk in mat files.
