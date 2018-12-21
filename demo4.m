%% Demo 4: Tucker decomposition of sparse tensor utilizing "double-sketch"
% 
% This script compares tucker_ts and tucker_ts_double_sketch. The example
% shows that the latter function, which incorporates the idea in 
% Remark 3.2 (c) of [2], is faster than the former in a situation when the
% dimension size I of a tensor is much larger than the sum of the target
% sketch dimensions J1+J2.
%
% The script is similar to demo1: It generates a sparse tensor and then 
% decomposes using both tucker_ts and tucker_ts_double_sketch. Since the 
% tensor in the example is quite large, we do not attempt to decompose it 
% tucker_als in Tensor Toolbox [1]. Please note that the script requires 
% Tensor Toolbox version 2.6 or later.
%
% REFERENCES:
%
%   [1] B. W. Bader, T. G. Kolda and others. MATLAB Tensor Toolbox 
%       Version 2.6, Available online, February 2015. 
%       URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%
%   [2] O. A. Malik, S. Becker. Low-Rank Tucker Decomposition of Large 
%       Tensors Using TensorSketch. Advances in Neural Information 
%       Processing Systems 32, pp. 10117-10127, 2018.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     December 21, 2018

%% Include relevant files

addpath(genpath('help_functions'));

%% Setup

R_true = [10 10 10]; % True tensor rank
R = [10 10 10]; % Algorithm target rank
I = [1e+6 1e+6 1e+6]; % Tensor size
K = 10; % Sketch dimension parameter
J1 = K*prod(R)/min(R); % First sketch dimension
J2 = K*prod(R); % Second sketch dimension
noise_level = 1e-3; % Amount of noise added to nonzero elements
tol = 1e-3; % Tolerance
maxiters = 50; % Maximum number of iterations
nnzY = 1e+6; % Approximate number of nonzeros in tensor

%% Generate random sparse tensor

fprintf('Generating sparse tensor... ');
N = length(I);
density = 1 - (1 - nnzY^(1/N)/I(1))^(1/R_true(1));
factor_mat_density = ones(length(R),1)*density;
[Y, ~, ~] = Generate_Random_Sptensor(R_true, I, factor_mat_density, noise_level);
fprintf('Done!\n\n');

%% Run tucker_ts

fprintf('\nRunning tucker_ts...\n\n')
tucker_ts_tic = tic;
[G_ts, A_ts] = tucker_ts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', true);
tucker_ts_toc = toc(tucker_ts_tic);

%% Run tucker_ts_double_sketch

fprintf('\nRunning tucker_ts_double_sketch...\n\n')
tucker_ts_double_sketch_tic = tic;
[G_tsds, A_tsds] = tucker_ts_double_sketch(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', true);
tucker_ts_double_sketch_toc = toc(tucker_ts_double_sketch_tic);

%% Results

fprintf('\n\nComputing errors... ');
normY = norm(Y);
tucker_ts_error = SptTtDiffNorm(Y, G_ts, A_ts)/normY;
tucker_tsds_error = SptTtDiffNorm(Y, G_tsds, A_tsds)/normY;
fprintf('Done!\n')

fprintf('\nRelative error for tucker_ts: %.6e\n', tucker_ts_error);
fprintf('Relative error for tucker_ttmts: %.6e\n', tucker_tsds_error);
fprintf('\nTime for tucker_ts: %.2f s\n', tucker_ts_toc);
fprintf('Time for tucker_ttmts: %.2f s\n', tucker_ts_double_sketch_toc);