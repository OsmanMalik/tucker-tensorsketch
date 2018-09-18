function [G, A] = tucker_ts(Y, R, J1, J2, varargin)
% TUCKER_TS     Implementation of one-pass TUCKER-TS algorithm. TUCKER-TS
%               utilizes TensorSketch to compute the Tucker decomposition
%               of a tensor.
%
%               This function requires Tensor Toolbox [1] version 2.6. 
% 
%   [G, A] = TUCKER_TS(Y, R, J1, J2) returns an approximate rank-R Tucker
%   decomposition of Y in the form of a core tensor G and factor matrices
%   A. Y can be a Matlab double array or one of the following types from 
%   Tensor Toolbox: ktensor, tensor, ttensor, or sptensor. R is a vector 
%   containing the target dimension, and J1 and J2 are the two sketch 
%   dimensions used. G is a Tensor Toolbox tensor, and A is a cell of 
%   matrices.
%
%   [G, A] = TUCKER_TS(input_cell, R, J1, J2) is the same as the last
%   example, but with the first variable changed to a cell called
%   input_cell. This option is used when we want to use an external
%   function for computing the sketches, e.g., when we want to load data
%   from the hard drive. input_cell should be of the following format:
%       - input_cell{1} should be a function handle pointing to a function 
%         which returns YsT and vecYs (see below); 
%       - input_cell{2} should be a vector containing the size of each
%         tensor dimension, e.g., I = [100 100 100] for a 3-way tensor of
%         size 100 x 100 x 100;
%       - the remaining entries of input_cell can be used to provide inputs
%         to the function in input_cell{1}.
%   The input function should take the following inputs, which are defined
%   below, in the order listed:
%       - Required: J1, J2, h1int64, h2int64, s, verbose.
%       - Optional inputs (provided via input_cell{3:end}).
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

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     September 17, 2018

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

%% Set extflag
% extflag will be true if we are using an external function to compute the
% sketches, otherwise it will be false

if iscell(Y)
    extflag = true;
else
    extflag = false;
end

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

if extflag
    sizeY = Y{2};
else
    sizeY = size(Y);
    nnzY = nnz(Y);
end
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

if verbose
    fprintf('Starting to compute sketches of input tensor...\n')
end

% Handle sparse tensor
if sflag 
    for n = 1:N
        if J1*sizeY(n) < 3*nnzY
            YsT{n} = SparseTensorSketchMatC_git(Y.vals, int64(Y.subs.'), h1int64(ns~=n), s(ns~=n), int64(J1), int64(sizeY(n)), int64(n));
        else
            % This case is for when the sketch dimension is so large that
            % storing YsT{n} in sparse format requires less memory 
            [subs, vals] = Sparse2SparseTensorSketchMatC_git(Y.vals, int64(Y.subs.'), h1int64(ns~=n), s(ns~=n), int64(J1), int64(sizeY(n)), int64(n));
            YsT{n} = sparse(subs(1,:), subs(2,:), vals, J1, sizeY(n));
        end
        if verbose
            fprintf('Finished computing sketch %d out of %d...\n', n, N+1)
        end
    end
    vecYs = SparseTensorSketchVecC_git(Y.vals, int64(Y.subs.'), h2int64, s, int64(J2));

% Use custom external function for computing sketches. Is e.g. used for
% handling dense large tensor in matfile in demo3.m provided with this
% software.
elseif extflag 
    sketch_func = Y{1};
    if length(Y) > 2
        sketch_params = {J1, J2, h1int64, h2int64, s, verbose, Y{3:end}};
    else
        sketch_params = {J1, J2, h1int64, h2int64, s, verbose};
    end
    [YsT, vecYs] = sketch_func(sketch_params{:});
    
% Handle dense normal tensor 
else 
    for n = 1:N
        YsT{n} = TensorSketchMatC3_git(double(tenmat(Y,n)), h1int64(ns~=n), s(ns~=n), int64(J1)).';
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
        % TensorSketch the Kronecker product and compute sketched LS problem
        A{n} = (ifft((double(tenmat(G,n)) * khatrirao(As1_hat(ns~=n), 'r')).') \ YsT{n}).';
            
        % Orthogonalize factor matrix and update core tensor
        [Qfac, Rfac] = qr(A{n},0);
        A{n} = Qfac;
        G = ttm(G, Rfac, n);
        
        % Update As1_hat{n}
        As1_hat{n} = fft(countSketch(A{n}.', h1int64{n}, J1, s{n}, 1), J1, 2);
    end
       
    % TensorSketch the Kronecker product using hash functions
    for n = 1:N
        As2_hat{n} = fft(countSketch(A{n}.', h2int64{n}, J2, s{n}, 1), J2, 2);
    end    
    
    % Compute sketched LS problem using conjugate gradient
    M2 = ifft(khatrirao(As2_hat, 'r').');
    M2tM2 = M2.'*M2;
    M2tvecYs = M2.'*vecYs;
    clear M2;
    myfun = @(x) M2tM2 * x; 
    G(:) = reshape(pcg(myfun, M2tvecYs), size(G));

    % Compute fit
    normG = norm(G);
    normChange = abs(normG - normG_old);
    fprintf(' Iter %2d: normChange = %7.1e\n', iter, normChange);
    
    % Check for convergence
    if (iter > 1) && (normChange < tol)
        break
    end
    
end

end