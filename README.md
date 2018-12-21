# Tucker-TensorSketch
Matlab functions for low-rank Tucker decomposition of tensors using TensorSketch.

## Some further details
Tucker-TensorSketch provides two functions, **tucker_ts** and **tucker_ttmts**, for low-rank Tucker decomposition of tensors. Both functions are variants of the standard alternating least-squares algorithm (higher-order orthogonal iteration) for the Tucker decomposition. They both incorporate a sketching technique called TensorSketch, which is a form of CountSketch that can be applied efficiently to matrices that are Kronecker products of smaller matrices. Due to the properties of TensorSketch, our functions only require a single pass of the input tensor. They can handle streamed data in the sense that they can read tensor elements in any order, and do not need to have access to all elements at the same time.

The figure below shows an experiment where we compare our functions to other competing methods. The experiment is done for sparse 3-way tensors each containing about 1e+6 nonzero elements. All sides of each tensor are of equal length. The plots show how the relative error (in subplot (a)) and run time (in subplot (b)) vary as the size of the tensor sides is increased. Our functions are fast and can handle larger tensors than competing methods, while still maintaining good accuracy. However, our functions do not scale well with the target rank, which is why we recommend using them for *low-rank* decompositions only.

![Experiment results](Experiment2Fig1.png)

For more details, please see our paper which is available here: http://papers.nips.cc/paper/8213-low-rank-tucker-decomposition-of-large-tensors-using-tensorsketch. Please use the following link to view the results of our video frame classification experiment in the paper: https://drive.google.com/open?id=1usBNBSfnPuy1S2Oy8-QQPusrvBmVTohl

The original video which we used to construct the dataset used in our video frame classification experiment is available for download here: https://drive.google.com/file/d/1HX6-motNPz_xZnkTJ32ZetRj_VUkI9XZ/view?usp=sharing. In our experiment, we converted this video to grayscale and then treated the video as a 3-way tensor. The video in the link is 173 MB in size. After converting it to grayscale and then simply treating it as an array of doubles its size is 38 GB. If you use this video in one your experiments, please include a reference to our paper.

## Requirements
This code requires Tensor Toolbox version 2.6 by Bader, Kolda and others (available at http://www.sandia.gov/~tgkolda/TensorToolbox/).

## Installation
Run the file **compile_all_mex.m** inside the folder help_functions. Alternatively, simply compile each c file individually by running "mex filename.c" inside Matlab.

## Demo files
The three demo files demonstrate our functions. Below is a brief description of each.
* **Demo 1:** This scrips gives a demo of tucker_ts and tucker_ttmts decomposing a sparse tensor. The script generates a sparse tensor and then decomposes it using both tucker_ts and tucker_ttmts, as well as tucker_als from Tensor Toolbox.
* **Demo 2:** This script gives a demo of tucker_ts and tucker_ttmts decomposing a dense tensor. The script generates a dense tensor and then decomposes it using both tucker_ts and tucker_ttmts, as well as tucker_als from Tensor Toolbox.
* **Demo 3:** This script gives a demo of tucker_ts and tucker_ttmts decomposing a dense tensor which is stored in a mat file on the hard drive. The result is compared to that produced by tucker_als in Tensor Toolbox applied to the same tensor stored in memory.

## Author contact information
Please feel free to contact me at any time if you have any questions or would like to provide feedback on this code or on our paper. I can be reached at osman.malik@colorado.edu.

## Licenses
This code uses the implementation of randomized SVD by Antoine Liutkus, which is available on MathWorks File Exchange. The license of that software is available in its original form in tucker-tensorsketch/help_functions/rsvd.

All other code in this project fall under the license in the root of this project.
