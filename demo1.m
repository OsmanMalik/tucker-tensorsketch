%% Demo 1: Tucker decomposition of sparse tensor
% 
% This scrips gives a demo of tucker_ts decomposing a sparse tensor. The
% script generates a sparse tensor and then decomposes it using both
% tucker_ts and tucker_als from Tensor Toolbox [1].
%
% REFERENCES:
%   [1]         Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor 
%               Toolbox Version 2.6, Available online, February 2015. URL: 
%               http://www.sandia.gov/~tgkolda/TensorToolbox/.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     May 24, 2018

%% Include relevant files
addpath(genpath('help_functions'));

%% Setup

R_true = [10 10 10]; % True tensor rank
R = [10 10 10]; % Algorithm target rank
I = [1e+3 1e+3 1e+3]; % Tensor size
K = 10; % Sketch dimension parameter
J1 = K*prod(R)/min(R); % First sketch dimension 
J2 = K*prod(R); % Second sketch dimension
noise_level = 1e-3; % Amount of noise added to nonzero elements
tol = 1e-3; % Tolerance
maxiters = 50; % Maximum number of iterations
nnzY = 1e+6; % Approximate number of nonzero in tensor

%% Generate random sparse tensor

fprintf('Generating sparse tensor... ');
N = length(I);
density = 1 - (1 - nnzY^(1/N)/I(1))^(1/R_true(1));
factor_mat_density = ones(4,1)*density;
[Y, ~, ~] = Generate_Random_Sptensor(R_true, I, factor_mat_density, noise_level);
fprintf('Done!\n\n');

%% Run tucker_ts

fprintf('\nRunning tucker_ts...\n\n')
tucker_ts_tic = tic;
[G, A] = tucker_ts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', true);
tucker_ts_toc = toc(tucker_ts_tic);

%% Run tucker_als

fprintf('\n\nRunning tucker_als...\n')
tucker_als_tic = tic;
Y_tucker_als = tucker_als(Y, R, 'tol', tol, 'maxiters', maxiters);
tucker_als_toc = toc(tucker_als_tic);

%% Results

fprintf('\n\nComputing errors... ');
normY = norm(Y);
tucker_ts_error = SptTtDiffNorm(Y, G, A)/normY;
tucker_als_error = SptTtDiffNorm(Y, Y_tucker_als.core, Y_tucker_als.U)/normY;
fprintf('Done!\n')

fprintf('\nRelative error for tucker_ts: %.6e\n', tucker_ts_error);
fprintf('Relative error for tucker_als: %.6e\n', tucker_als_error);
fprintf('tucker_ts error relative to tucker_als error: %.2f\n', tucker_ts_error/tucker_als_error);
fprintf('\nTime for tucker_ts: %.2f s\n', tucker_ts_toc);
fprintf('Time for tucker_als: %.2f s\n', tucker_als_toc);