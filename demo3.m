%% Demo 3: Tucker decomposition of dense tensor in mat file
%
% This script gives a demo of tucker_ts decomposing a dense tensor which is
% stored in a mat file. The result is compared to that produced by
% tucker_als in Tensor Toolbox [1] applied to the same tensor but in
% memory. Note that the size of the tensor is limited to allow comparison
% with tucker_als. This is not necessary for tucker_ts, and the size can be
% increased if the code for tucker_als is commented out.
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
inc_size = [100 100 100];
filename = 'test.mat';

%% Generate random dense tensor and save it to mat file

fprintf('Generating dense tensor... ');
G_true = tensor(randn(R_true));
A_true = cell(length(R_true),1);
for k = 1:length(R_true)
    A_true{k} = randn(I(k),R_true(k));
    [Qfac, ~] = qr(A_true{k}, 0);
    A_true{k} = Qfac;
end
fprintf('Done!\n\n');

fprintf('Creating matfile...\n');
file = matfile(filename, 'Writable', true);
file.Y = nan(2,2,2);
for i = 1:I(3)/inc_size(3)
    Gai = ttm(G_true, A_true{3}(1+(i-1)*inc_size : i*inc_size, :), 3);
    file.Y(1:I(1), 1:I(2), 1+(i-1)*inc_size : i*inc_size) ...
        = double(ttm(Gai, A_true(1:2), [1 2])) ...
        + noise_level*randn([I(1:end-1) inc_size(3)]);
    fprintf('\t%.0f%%\n', i*inc_size(3)/I(3)*100);
end
fprintf('\tDone!\n\n');

%% Run tucker_ts using mat file as input

fprintf('\nRunning tucker_ts...\n\n')
tucker_ts_tic = tic;
[G, A] = tucker_ts({filename, inc_size}, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', true);
tucker_ts_toc = toc(tucker_ts_tic);

%% Results

fprintf('\n\nComputing errors...\n');
normY = 0;
normDiff_tucker_ts = 0;
for i = 1:I(1)/inc_size(1)
    slice_start = 1+(i-1)*inc_size(1);
    slice_end = i*inc_size(1);
    Y_piece = tensor(file.Y(slice_start:slice_end , :, :));
    
    Y_tucker_ts_piece = tensor(ttensor(G, {A{1}(slice_start:slice_end, :), A{2:end}} ));
    normDiff_tucker_ts = normDiff_tucker_ts + norm(Y_piece - Y_tucker_ts_piece)^2;
    clear Y_tucker_ts_piece
    
    normY = normY + norm(Y_piece)^2;
    fprintf('\t%.2f%%\n', i*inc_size(1)/I(1)*100);
end
normDiff_tucker_ts = sqrt(normDiff_tucker_ts);
normY = sqrt(normY);
fprintf('\tDone!\n\n');

fprintf('Relative error for tucker_ts: %.6e\n', normDiff_tucker_ts/normY);
fprintf('Time for tucker_ts: %.2f s\n', tucker_ts_toc);
