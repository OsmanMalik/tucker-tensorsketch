%% Demo 3: Tucker decomposition of dense tensor in mat file
%
% This script gives a demo of tucker_ts and tucker_ttmts decomposing a
% dense tensor which is stored in a mat file. The result is compared to
% that produced by tucker_als in Tensor Toolbox [1] applied to the same 
% tensor stored in memory. Note that the size of the tensor is limited to 
% allow comparison with tucker_als. This is not necessary for tucker_ts and
% tucker_ttmts, and the size of the tensor can be increased if the code for
% tucker_als is commented out. Please note that the script requires Tensor
% Toolbox version 2.6 or later.
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
%       Processing Systems 32, pp. 10117-10127, 2018.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     December 21, 2018

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
inc_size = [100 100 100]; % Increment size in each dimension when saving 
                          % tensor to mat file, computing errors, etc. Each
                          % number must divide the corresponding dimension
                          % size.
filename = 'demo3_tensor.mat'; % Name of tensor mat file

%% Generate random dense tensor and save it to mat file

fprintf('Generating dense tensor... ');
G_true = tensor(randn(R_true));
A_true = cell(length(R_true),1);
for k = 1:length(R_true)
    A_true{k} = randn(I(k),R_true(k));
    [Qfac, ~] = qr(A_true{k}, 0);
    A_true{k} = Qfac;
end
noise = noise_level*randn(I);
fprintf('Done!\n\n');

fprintf('Creating matfile...\n');
file = matfile(filename, 'Writable', true);
file.Y = nan(2,2,2);
for i = 1:I(3)/inc_size(3)
    Gai = ttm(G_true, A_true{3}(1+(i-1)*inc_size(3) : i*inc_size(3), :), 3);
    file.Y(1:I(1), 1:I(2), 1+(i-1)*inc_size(3) : i*inc_size(3)) ...
        = double(ttm(Gai, A_true(1:2), [1 2])) ...
        + noise(:, :, 1+(i-1)*inc_size(3):i*inc_size(3));
    fprintf('\t%.0f%%\n', i*inc_size(3)/I(3)*100);
end
fprintf('\tDone!\n\n');

%% Run tucker_ts using mat file as input

fprintf('\nRunning tucker_ts...\n\n')
tucker_ts_tic = tic;
inpt = {@sketch_from_mat_ts, I, filename, inc_size};
[G_ts, A_ts] = tucker_ts(inpt, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', true);
tucker_ts_toc = toc(tucker_ts_tic);

%% Run tucker_ttmts using mat file as input

fprintf('\nRunning tucker_ttmts...\n\n')
tucker_ttmts_tic = tic;
inpt = {@sketch_from_mat_ttmts, I, filename, inc_size};
[G_ttmts, A_ttmts] = tucker_ttmts(inpt, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', true);
tucker_ttmts_toc = toc(tucker_ttmts_tic);

%% Run tucker_als

Y = tensor(ttensor(G_true, A_true)) + noise;
fprintf('\n\nRunning tucker_als...\n')
tucker_als_tic = tic;
Y_tucker_als = tucker_als(Y, R, 'tol', tol, 'maxiters', maxiters);
tucker_als_toc = toc(tucker_als_tic);

%% Results

fprintf('\n\nComputing errors...\n');
normY = 0;
normDiff_tucker_ts = 0;
normDiff_tucker_ttmts = 0;
for i = 1:I(1)/inc_size(1)
    slice_start = 1+(i-1)*inc_size(1);
    slice_end = i*inc_size(1);
    Y_piece = tensor(file.Y(slice_start:slice_end , :, :));
    
    Y_tucker_ts_piece = tensor(ttensor(G_ts, {A_ts{1}(slice_start:slice_end, :), A_ts{2:end}} ));
    normDiff_tucker_ts = normDiff_tucker_ts + norm(Y_piece - Y_tucker_ts_piece)^2;
    Y_tucker_ttmts_piece = tensor(ttensor(G_ttmts, {A_ttmts{1}(slice_start:slice_end, :), A_ttmts{2:end}} ));
    normDiff_tucker_ttmts = normDiff_tucker_ttmts + norm(Y_piece - Y_tucker_ttmts_piece)^2;
    clear Y_tucker_ts_piece
    
    normY = normY + norm(Y_piece)^2;
    fprintf('\t%.2f%%\n', i*inc_size(1)/I(1)*100);
end
normDiff_tucker_ts = sqrt(normDiff_tucker_ts);
normDiff_tucker_ttmts = sqrt(normDiff_tucker_ttmts);
normY = sqrt(normY);
tucker_ts_error = normDiff_tucker_ts/normY;
tucker_ttmts_error = normDiff_tucker_ttmts/normY;
tucker_als_error = norm(Y - tensor(Y_tucker_als))/normY;
fprintf('\tDone!\n\n');

fprintf('Relative error for tucker_ts: %.6e\n', tucker_ts_error);
fprintf('Relative error for tucker_ttmts: %.6e\n', tucker_ttmts_error);
fprintf('Relative error for tucker_als: %.6e\n', tucker_als_error);
fprintf('tucker_ts error relative to tucker_als error: %.2f\n', tucker_ts_error/tucker_als_error);
fprintf('tucker_ttmts error relative to tucker_als error: %.2f\n', tucker_ttmts_error/tucker_als_error);
fprintf('\nTime for tucker_ts: %.2f s\n', tucker_ts_toc);
fprintf('Time for tucker_ttmts: %.2f s\n', tucker_ttmts_toc);
