function [G, A] = tucker_ts_double_sketch(Y, R, J1, J2, varargin)
% TUCKER_TS_DOUBLE_SKETCH   Implementation of one-pass TUCKER-TS algorithm.
%                           TUCKER-TS utilizes TensorSketch to compute the
%                           Tucker decomposition of a tensor. Reduces size
%                           of least-squares problem by utilizing
%                           "double-sketching"; specifically, it uses the
%                           idea in Remark 3.2 (c) of [2].
%
%                           This function requires Tensor Toolbox [1] 
%                           version 2.6.
% 
%   [G, A] = TUCKER_TS_DOUBLE_SKETCH(Y, R, J1, J2) returns an approximate
%   rank-R Tucker decomposition of Y in the form of a core tensor G and 
%   factor matrices A. Y can be a Matlab double array or one of the 
%   following types from Tensor Toolbox: ktensor, tensor, ttensor, or 
%   sptensor. R is a vector containing the target dimension, and J1 and J2
%   are the two sketch dimensions used. G is a Tensor Toolbox tensor, and A 
%   is a cell of matrices.
%
%   [G, A] = TUCKER_TS(___, Name, Value) specifies optional parameters
%   using one or more Name, Value pair arguments. The following optional
%   parameters are available: 'tol' sets the tolerance (default 1e-3),
%   'maxiters' sets the maximum number of iterations (default 50),
%   'verbose' controls how much information is printed during execution
%   (true or false; default false).
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

%% Handle optional inputs

params = inputParser;
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'verbose', false, @isscalar);
parse(params, varargin{:});

tol = params.Results.tol;
maxiters = params.Results.maxiters;
verbose = params.Results.verbose;

%% Convert input tensor to double array if necessary. Set sflag (sparsity flag)

sflag = false;
if ismember(class(Y), {'ktensor', 'tensor', 'ttensor'})
    Y = double(Y);
elseif ismember(class(Y), {'sptensor'})
    sflag = true;
elseif ~ismember(class(Y), {'double', 'cell'})
    error('ERROR: Invalid format for Y.')
end

%% Define hash functions

sizeY = size(Y);
N = length(sizeY);

h1int64 = cell(N, 1);
h2int64 = cell(N, 1);
s = cell(N, 1);
for n = 1:N
    h1int64{n} = int64(randi(J1, sizeY(n), 1));
    h2int64{n} = int64(randi(J2, sizeY(n), 1));
    s{n} = round(rand(sizeY(n), 1))*2 - 1;
end

%% Initialize factor matrices and core tensor

A = cell(N,1);
As1_hat = cell(N,1);
As2_hat = cell(N,1);
Rs2 = cell(N,1);
G = tensor(rand(R)*2-1);
for n = 2:N
    A{n} = rand(sizeY(n),R(n))*2-1;
    [Qfac,~] = qr(A{n},0);
    A{n} = Qfac;
    As1_hat{n} = fft(countSketch(A{n}.', h1int64{n}, J1, s{n}, 1), J1, 2);
end

%% Compute a total of N+1 TensorSketches of different shapes of Y

ns = 1:N;
YsT = cell(N, 1);
YsT_ds1 = cell(N, 1);
YsT_ds2 = cell(N, 1);

if verbose
    fprintf('Starting to compute sketches of input tensor...\n')
end

% Handle sparse tensor
if sflag 
    for n = 1:N
        [YsT{n}, YsT_ds1{n}, YsT_ds2{n}] = SparseTensorSketchMatDSC(Y.vals, int64(Y.subs.'), h1int64(ns~=n), h1int64{n}, h2int64{n}, s(ns~=n), s{n}, int64(J1), int64(J2), int64(sizeY(n)), int64(n));
        if verbose
            fprintf('Finished computing sketch %d out of %d...\n', n, N+1)
        end
    end
    vecYs = SparseTensorSketchVecC_git(Y.vals, int64(Y.subs.'), h2int64, s, int64(J2));
  
% Handle dense normal tensor 
else 
    for n = 1:N
        YsT{n} = TensorSketchMatC3_git(double(tenmat(Y,n)), h1int64(ns~=n), s(ns~=n), int64(J1)).';
        YsT_ds1{n} = countSketch(YsT{n}, h1int64{n}, J1, s{n}, 1);
        YsT_ds2{n} = countSketch(YsT{n}, h1int64{n}, J2, s{n}, 1);
        if verbose
            fprintf('Finished computing sketch %d out of %d...\n', n, N+1)
        end
    end
    vecYs = TensorSketchVecC_git(Y, h2int64, s, int64(J2));
end

if verbose
    fprintf('Finished computing all sketches\n')
end

clear Y

%% Main loop: Iterate until convergence, for a maximum of maxiters iterations

if verbose
    fprintf('Starting main loop...\n')
end

normG = norm(G);
for iter = 1:maxiters
    
    normG_old = normG;
    
    for n = 1:N
        % TensorSketch the Kronecker product and compute double-sketched LS
        % problem
        M1 = ifft(khatrirao(As1_hat(ns~=n), 'r').');
        SOL = M1*double(tenmat(G,n)).' \ [YsT_ds1{n}, YsT_ds2{n}];
        
        % Update As1_hat{n} and As2_hat{n} 
        As1_hat{n} = fft(SOL(:,1:J1), J1, 2);
        As2_hat{n} = fft(SOL(:,J1+1:end), J2, 2);
        
        % Compute Rs2{n} which is used in stopping condition
        [~, Rs2{n}] = qr(SOL(:,J1+1:end).', 0);
                    
    end
    
    % Compute core tensor
    M2 = ifft(khatrirao(As2_hat, 'r').');
    G(:) = reshape(M2 \ vecYs, size(G));
    
    % Compute fit using heuristic described in supplement of [2]
    G_normalized = ttm(G, Rs2);
    normG = norm(G_normalized);
    normChange = abs(normG - normG_old);
    fprintf(' Iter %2d: normChange = %7.1e\n', iter, normChange);
    
    % Check for convergence
    if (iter > 1) && (normChange < tol)
        break
    end
    
end

% Compute full factor matrices
for n = 1:N
    % TensorSketch the Kronecker product using hash functions
    M1 = ifft(khatrirao(As1_hat(ns~=n), 'r').');

    % Compute sketched LS problem
    A{n} = (M1*double(tenmat(G,n)).' \ YsT{n}).';
    As1_hat{n} = fft(countSketch(A{n}.', h1int64{n}, J1, s{n}, 1), J1, 2);
end

% Compute core tensor one final time based on new factor matrices
for n = 1:N
    As2_hat{n} = fft(countSketch(A{n}.', h2int64{n}, J2, s{n}, 1), J2, 2);
end
M2 = ifft(khatrirao(As2_hat, 'r').');
G(:) = reshape(M2 \ vecYs, size(G));

% Orthogonalize factor matrix and update core tensor accordingly
for n = 1:N
    [Qfac, Rfac] = qr(A{n},0);
    A{n} = Qfac;
    G = ttm(G, Rfac, n);
end

end