%% TensorSketch verification
%
% The purpose of this script is to verify that the formula in Eq (5) of our
% paper [1]---which avoids forming the full Kronecker product---yields the 
% same output as our codes that do this computation without this trick and
% instead treats TensorSketch as a big CountSketch. Please note that the 
% script requires Tensor Toolbox [2] version 2.6 or later.
% 
% Please see the supplement of our paper for a mathematical derivation
% of Eq (5).
%
% Since the purpose of this script is pedagogical, the implementation of Eq
% (5) in [2] that we do here in this script is not efficient. Please see
% our code in e.g. tucker_ts.m for a more efficient implementation which
% leverages C code for computing the CountSketches of the smaller
% matrices/vectors. 
%
% REFERENCES:
%
%   [1] O. A. Malik, S. Becker. Low-Rank Tucker Decomposition of Large 
%       Tensors Using TensorSketch. Advances in Neural Information 
%       Processing Systems (NeurIPS), 2018.
%
%   [2] B. W. Bader, T. G. Kolda and others. MATLAB Tensor Toolbox 
%       Version 2.6, Available online, February 2015. 
%       URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.

clear all
addpath(genpath('help_functions'));

% Settings
N = 3; % Number of matrices/vectors in Kronecker product
m = 20; % Number of rows in each submatrix
n = 2; % Number of columns in each submatrix (n = 1 for vector)
J = 2000; % Target sketch dimension

% Note: If choosing e.g. N=3, m=20, n=2 and J=2000 above, then the large 
% matrix will be 20^3 by 2^3 in size, and the sketched matrix will be 2000
% by 2^3 in size.

% Create N submatrices/subvectors
A_sub = cell(1,N); 
for k = 1:N
    A_sub{k} = randn(m,n);
end

% Create large matrix/vector from submatrices/subvectors.
% Notice that this is done in reverse in order to match our paper.
A = kron(A_sub{end}, A_sub{end-1});
for k = 2:N-1
    A = kron(A, A_sub{end-k});
end

% Create sign vectors, hash functions, sketch matrices
s = cell(1,N); % Cell for sign vectors
h = cell(1,N); % Cell for hash functions
sketch_mat = cell(1,N);
h_int64 = cell(1,N);
for k = 1:N
    s{k} = zeros(J, m);
    h{k} = randi(J, m, 1);
    h_int64{k} = int64(h{k});
    s{k} = (round(rand(m,1))*2-1);   
    sketch_mat{k} = zeros(J, m);
    sketch_mat{k}(h{k} + J*(0:m-1).') = s{k}; % CountSketch matrices
end

% Compute TensorSketch via Eq (5) in our paper
SAk = cell(1,N);
for k = 1:N
    SAk{k} = fft(sketch_mat{k}*A_sub{k}).';
end
krSAk = khatrirao(SAk, 'r'); % The added 'r' does the product in reverse
TA_1 = ifft(krSAk.'); % This is the sketched matrix/vector!

% Compute TensorSketch by (inefficiently) applying large CountSketch
% corresponding to TensorSketch. Ty_2 is the sketched matrix/vector!
if n == 1
    fprintf('Using vector sketching...\n')
    TA_2 = TensorSketchVecC_git(A, h_int64, s, int64(J));
else
    fprintf('Using matrix sketching...\n')
    TA_2 = TensorSketchMatC3_git(A.', h_int64, s, int64(J)).';
end

% Compute discrepancy --- should be close to zero since mathematically the
% two computations are the same
er = norm(TA_1 - TA_2, 'fro');
fprintf('Difference between two computations is %.4e\n\n', er);
