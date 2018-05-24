%% Demo 2: Tucker decomposition of dense tensor
%
% This script gives a demo of tucker_ts decomposing a dense tensor. The
% script generates a dense tensor and then decomposes it using both
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
I = [400 400 400]; % Tensor size
K = 10; % Sketch dimension parameter
J1 = K*prod(R)/min(R); % First sketch dimension 
J2 = K*prod(R); % Second sketch dimension
noise_level = 1e-3; % Amount of noise added to nonzero elements
tol = 1e-3; % Tolerance
maxiters = 50; % Maximum number of iterations

%% Generate random dense tensor

fprintf('Generating dense tensor... ');
G_true = tensor(randn(R_true));
A_true = cell(length(R_true),1);
for k = 1:length(R_true)
    A_true{k} = randn(I(k),R_true(k));
    [Qfac, ~] = qr(A_true{k}, 0);
    A_true{k} = Qfac;
end
Y = tensor(ttensor(G_true, A_true)) + noise_level*randn(I);
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
tucker_ts_error = norm(Y - tensor(ttensor(G,A)))/normY;
tucker_als_error = norm(Y - tensor(Y_tucker_als))/normY;
fprintf('Done!\n')

fprintf('\nRelative error for tucker_ts: %.6e\n', tucker_ts_error);
fprintf('Relative error for tucker_als: %.6e\n', tucker_als_error);
fprintf('tucker_ts error relative to tucker_als error: %.2f\n', tucker_ts_error/tucker_als_error);
fprintf('\nTime for tucker_ts: %.2f s\n', tucker_ts_toc);
fprintf('Time for tucker_als: %.2f s\n', tucker_als_toc);