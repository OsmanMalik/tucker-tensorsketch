# Tucker-TensorSketch
Tucker-TensorSketch provides Matlab functions for low-rank Tucker decomposition of tensors using TensorSketch. For further information about our methods, please see our paper:

O. A. Malik and S. Becker. Low-Rank Tucker Decomposition of Large Tensors Using TensorSketch. *Advances in Neural Information Processing Systems 32*, pages 10117-10127, 2018.

It is available at http://papers.nips.cc/paper/8213-low-rank-tucker-decomposition-of-large-tensors-using-tensorsketch.

## Some further details
Tucker-TensorSketch provides three functions, **tucker_ts**, **tucker_ts_double_sketch** and **tucker_ttmts**, for low-rank Tucker decomposition of tensors. These functions are variants of the standard alternating least-squares algorithm (higher-order orthogonal iteration) for the Tucker decomposition. They incorporate a sketching technique called TensorSketch, which is a form of CountSketch that can be applied efficiently to matrices which are Kronecker products of smaller matrices. Due to the properties of TensorSketch, our functions only require a single pass of the input tensor. They can handle streamed data in the sense that they can read tensor elements in any order, and do not need to have access to all elements at the same time. The functions **tucker_ts** and **tucker_ttmts** incorporate sketching in different ways. The function **tucker_ts_double_sketch** is a variant of **tucker_ts** which incorporates the idea presented in Remark 3.2 (c) of our paper.

The figure below shows an experiment where we compare our functions to other competing methods. The experiment is done for sparse 3-way tensors each containing about 1e+6 nonzero elements. All sides of each tensor are of equal length. The plots show how the relative error (in subplot (a)) and run time (in subplot (b)) vary as the size of the tensor sides is increased. Our functions are fast and can handle larger tensors than competing methods, while still maintaining good accuracy. However, our functions do not scale well with the target rank, which is why we recommend using them for *low-rank* decomposition only.

![Experiment results](Experiment2Fig1.png)

## Video experiment
In our paper, we present an experiment where we apply one of our methods to a 3-way tensor representing a video and then use the resulting decomposition to classify frames that contain a distubance. Use the following link to view a video showing the results of this experiment: https://drive.google.com/open?id=1usBNBSfnPuy1S2Oy8-QQPusrvBmVTohl

The original video which we used in this experiment is available for download here: https://drive.google.com/file/d/1HX6-motNPz_xZnkTJ32ZetRj_VUkI9XZ/view?usp=sharing. In our experiment, we converted this video to grayscale and then treated the video as a 3-way tensor. The video in the link is 173 MB in size. After converting it to grayscale and then simply treating it as an array of doubles its size is about 38 GB. Feel free to use this video in your own experiments. If you do, please provide a reference to our paper.

## Requirements
Our code requires Tensor Toolbox version 2.6 by Bader, Kolda and others (available at http://www.sandia.gov/~tgkolda/TensorToolbox/).

## Installation
Run the file **compile_all_mex.m** inside the folder help_functions. Alternatively, simply compile each C file individually by running "mex filename.c" inside Matlab.

## Demo files
The four demo files demonstrate our functions. Below is a brief description of each.
* **Demo 1:** This scrips gives a demo of **tucker_ts** and **tucker_ttmts** decomposing a sparse tensor. The script generates a sparse tensor and then decomposes it using both **tucker_ts** and **tucker_ttmts**, as well as **tucker_als** from Tensor Toolbox.
* **Demo 2:** This script gives a demo of **tucker_ts** and **tucker_ttmts** decomposing a dense tensor. The script generates a dense tensor and then decomposes it using both **tucker_ts** and **tucker_ttmts**, as well as **tucker_als** from Tensor Toolbox.
* **Demo 3:** This script gives a demo of **tucker_ts** and **tucker_ttmts** decomposing a dense tensor which is stored in a mat file on the hard drive. The result is compared to that produced by **tucker_als** in Tensor Toolbox applied to the same tensor stored in memory.
* **Demo 4:** This script gives a demo of **tucker_ts** and **tucker_ts_double_sketch** decomposing a sparse tensor. The script generates a sparse tensor and then decomposes it using both **tucker_ts** and **tucker_ts_double_sketch**. The idea is to demonstrate that the technique which we present in Remark 3.2 (c) of our paper improves the speed of the TUCKER-TS algorithm when the tensor dimension sizes are much larger than the sum of the two target sketch dimensions.

## Referencing this code
If you use our code in any of your own work, please reference our paper:
```
@inproceedings{Malik-Becker-2018,
  author    = {Osman Asif Malik and Stephen Becker},
  title     = {Low-Rank {Tucker} Decomposition of Large Tensors Using {TensorSketch}},
  booktitle = {Advances in {Neural} {Information} {Processing} {Systems} 32},
  pages     = {10096--10106},
  year      = {2018},
}
```

## Author contact information
Please feel free to contact me at any time if you have any questions or would like to provide feedback on this code or on our paper. I can be reached at osman.malik@colorado.edu.

## Licenses
This code uses the implementation of randomized SVD by Antoine Liutkus, which is available on MathWorks File Exchange. The license of that software is available in its original form in tucker-tensorsketch/help_functions/rsvd.

All other code in this project falls under the license in the root of this project.
