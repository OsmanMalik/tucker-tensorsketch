function [G, A] = tucker_ts(Y, R, J1, J2, varargin)
% tucker_ts     Implementation of one-pass TUCKER-TS algorithm. TUCKER-TS
%               utilizes TensorSketch to compute the Tucker decomposition
%               of a tensor.
%
%               This function requires Tensor Toolbox [1] version 2.6. 
% 
%   [G, A] = TUCKER_TS(Y, R, J1, J2) returns an approximate rank-R Tucker
%   decomposition of Y in the form of a core tensor G and factor matrices
%   A. Y can be a Matlab double array or one of the following types from 
%   Tensor Toolbox: ktensor, tensor, ttensor, or sptensor. 
%   Alternatively, Y can be a cell of the form 
%   {'matfilename.mat',  inc_size}, where matfilename is the name of a
%   mat file containing a tensor in the form of a Matlab double array, and 
%   inc_size is a vector that contains the increment size to be used when
%   used when reading in the mat file tensor (must divide the corresponding
%   tensor dimension size). R is a vector containing the target dimension,
%   and J1 and J2 are the two sketch dimensions used. G is a Tensor Toolbox
%   tensor, and A is a cell of matrices.
%
%   [G, A] = TUCKER_TS(___, Name, Value) specifies optional parameters
%   using one or more Name, Value pair arguments. The following optional
%   parameters are available: 'tol' sets the tolerance (default 1e-3),
%   'maxiters' sets the maximum number of iterations (default 50),
%   'verbose' controls how much information is printed during execution
%   (default false).
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

%% Handle optional inputs
params = inputParser;
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'verbose', false, @isscalar);
parse(params, varargin{:});

tol = params.Results.tol;
maxiters = params.Results.maxiters;
verbose = params.Results.verbose;

%% Set 'piece-by-piece' flag pbpflag
% pbpflag will be true if we are reading a tensor from a mat file
if iscell(Y)
    pbpflag = true;
    file = matfile(Y{1}, 'Writable', false);
    inc_size = Y{2};
else
    pbpflag = false;
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
if pbpflag
    sizeY = size(file, 'Y');
else
    sizeY = size(Y);
    nnzY = nnz(Y);
end
N = length(sizeY);

h1int64 = cell(N,1);
h2int64 = cell(N,1);
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
        if J1*sizeY(n) < (N+1)*nnzY
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

% Handle dense large tensor in matfile
elseif pbpflag 
    for n = 1:N
        if n < N
            no_inc = sizeY(n)/inc_size(n);
        else
            no_inc = sizeY(N-1)/inc_size(n);
        end
        if rem(no_inc,1) ~= 0
            error('ERROR: The increment size must divide the relevant tensor dimension sizes.')
        end
        YsT{n} = zeros(J1, sizeY(n));
        if n == 1
            vecYs = zeros(J2, 1);
        end
        for inc = 1:no_inc
            slice_start = 1+(inc-1)*inc_size(n);
            slice_end = inc*inc_size(n);
            if n < N
                colons = repmat({':'}, 1, N-1);
                Y_piece = file.Y(colons{:}, slice_start : slice_end);
            else % n == N
                colons = repmat({':'}, 1, N-2);
                Y_piece = file.Y(colons{:}, slice_start : slice_end, :);
            end
            YsT{n} = YsT{n} + TensorSketchMatC3_git(double(tenmat(Y_piece, n)), h1int64(ns~=n), s(ns~=n), ...
                int64(J1), int64(slice_start), int64(slice_end)).';
            clear Y_piece_n
            if n == 1
                vecYs = vecYs + TensorSketchVecC_git(Y_piece, h2int64, s, ...
                    int64(J2), int64(slice_start), int64(slice_end));
            end
        end
        if verbose
            fprintf('Finished computing sketch %d out of %d...\n', n+1, N+1)
        end
    end
    
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

%% Main loop: Iterate until convergence, for a maximum of no_repeat iterations
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