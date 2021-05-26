%% Demo 1: Tucker decomposition of sparse tensor
% 
% This scrips gives a demo of tucker_ts and tucker_ttmts decomposing a
% sparse tensor. The script generates a sparse tensor and then decomposes
% it using both tucker_ts and tucker_ttmts, as well as tucker_als from
% Tensor Toolbox [1]. Please note that the script requires Tensor Toolbox
% version 2.6 or later.
%
% For further information about our methods, please see our paper [2].
%
% REFERENCES:
%
%   [1] B. W. Bader, T. G. Kolda and others. MATLAB Tensor Toolbox 
%       Version 2.6, Available online, February 2015. 
%       URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%
%   [2] O. A. Malik, S. Becker. Low-Rank Tucker Decomposition of Large 
%       Tensors Using TensorSketch. Advances in Neural Information 
%       Processing Systems (NeurIPS), 2018.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     December 21, 2018

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

%% Run tucker_ttmts

fprintf('\nRunning tucker_ttmts...\n\n')
tucker_ttmts_tic = tic;
[G_ttmts, A_ttmts] = tucker_ttmts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', true);
tucker_ttmts_toc = toc(tucker_ttmts_tic);

%% Run tucker_als

fprintf('\n\nRunning tucker_als...\n')
tucker_als_tic = tic;
Y_tucker_als = tucker_als(Y, R, 'tol', tol, 'maxiters', maxiters);
tucker_als_toc = toc(tucker_als_tic);

%% Results

fprintf('\n\nComputing errors... ');
normY = norm(Y);
tucker_ts_error = SptTtDiffNorm(Y, G_ts, A_ts)/normY;
tucker_ttmts_error = SptTtDiffNorm(Y, G_ttmts, A_ttmts)/normY;
tucker_als_error = SptTtDiffNorm(Y, Y_tucker_als.core, Y_tucker_als.U)/normY;
fprintf('Done!\n')

fprintf('\nRelative error for tucker_ts: %.6e\n', tucker_ts_error);
fprintf('Relative error for tucker_ttmts: %.6e\n', tucker_ttmts_error);
fprintf('Relative error for tucker_als: %.6e\n', tucker_als_error);
fprintf('tucker_ts error relative to tucker_als error: %.2f\n', tucker_ts_error/tucker_als_error);
fprintf('tucker_ttmts error relative to tucker_als error: %.2f\n', tucker_ttmts_error/tucker_als_error);
fprintf('\nTime for tucker_ts: %.2f s\n', tucker_ts_toc);
fprintf('Time for tucker_ttmts: %.2f s\n', tucker_ttmts_toc);
fprintf('Time for tucker_als: %.2f s\n', tucker_als_toc);