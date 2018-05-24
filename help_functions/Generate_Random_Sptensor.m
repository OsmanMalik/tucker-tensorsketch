function [Y, G, A] = Generate_Random_Sptensor(R, I, densities, noise_level)
% GENERATE_RANDOM_SPTENSOR  Generate random sparse tensor
%
%   [Y, G, A] = GENERATE_RANDOM_SPTENSOR(R, I, densities, noise_level)
%   returns a sparse tensor Y and the corresponding core tensor G and
%   factor matrices A. Y is an sptensor from Tensor Toolbox [1], G is a
%   tensor from Tensor Toolbox, and A is a cell of matrices. The inputs are
%   R, a vector containing the true matrix rank; I, a vector containing the
%   tensor dimension sizes; densities, a vector containing the densities of
%   each true factor matrix; and noise_level, a scalar which specifices the
%   level of noise to be added to each nonzero element of Y.
%
% REFERENCES:
%   [1]         Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor 
%               Toolbox Version 2.6, Available online, February 2015. URL: 
%               http://www.sandia.gov/~tgkolda/TensorToolbox/.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     May 24, 2018

% Initialize core tensor and factor matrices
G = tensor(randn(R));
A = cell(length(R), 1);
for k = 1:length(R)
    A{k} = sprand(I(k), R(k), densities(k));
    nnz_idx = find(A{k});
    A{k}(nnz_idx) = A{k}(nnz_idx)*2-1;
    [Qfac, ~] = qr(A{k}, 0);
    A{k} = Qfac;
end

% Create sparse tensor using CreateSparseTensor.c, and add noise
Y = CreateSparseTensor(G.data, A);
nnz_idx = find(Y);
Y(nnz_idx) = Y(nnz_idx) + noise_level*randn(size(nnz_idx,1),1);

end