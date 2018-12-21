function X = CreateSparseTensor(G, A)
% CREATESPARSETENSOR    This is a wrapper function to the C function
%                       CreateSparseTensorC. This wrapper calls that C
%                       function, and also prepares the core tensor G and
%                       cell of sparse factor matrices A for input into it.
%
%   X = GENERATE_RANDOM_SPTENSOR(G, A) returns a Tensor Toolbox [1]
%   sptensor constructed from the core tensor G and factor matrices A. G
%   must be a Matlab double array, and A must be a cell containing the
%   factor matrices. These matrices should be sparse Matlab double arrays.
%
% REFERENCES:
%
%   [1] Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor 
%       Toolbox Version 2.6, Available online, February 2015. URL: 
%       http://www.sandia.gov/~tgkolda/TensorToolbox/.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     December 21, 2018

Avals = cell(length(A),1);
Asubs = cell(length(A),1);
no_output_vals = 1;
R_true = zeros(1,length(A));
I = zeros(1,length(A));
for k = 1:length(A)
    [r, c, v] = find(A{k});
    [tmp, sort_idx] = sortrows([r c]);
    Asubs{k} = int32(tmp.');
    Avals{k} = v(sort_idx);    
    no_output_vals = no_output_vals*nnz(sum(A{k},2));
    R_true(k) = size(A{k},2);
    I(k) = size(A{k},1);
end
[Xsubs, Xvals] = CreateSparseTensorC(Asubs, Avals, G, int32(R_true), int32(no_output_vals));
X = sptensor(double(Xsubs).', Xvals, I);

end