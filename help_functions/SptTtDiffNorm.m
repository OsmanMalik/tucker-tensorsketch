function nrm = SptTtDiffNorm(X, G, A)
% SPTTTDIFFNORM     Compute the norm ||X-[G;A]||_F where X is an sptensor
%                   and [G;A] is a Tucker tensor defined by the core tensor
%                   G and factor matrices in A.
%
%   nrm = SPTTTDIFFNORM(X, G, A) returns the norm ||X-[G;A]||_F, where X is
%   an sptensor and [G;A] is a Tucker tensor defined by the core tensor G 
%   and factor matrices in A. X needs to be a Tensor Toolbox [1] sptensor,
%   G must be a Tensor Toolbox tensor, and A must be a cell containing the
%   factor matrices, which should be Matlab dense double arrays.
%
% NOTES:
%   Note 1  - Due to the way the norm is computed---by squaring,
%             subtracting, and then taking the square root---there is a
%             substantial loss of accuracy. Based on experimentation, it
%             seems like this function is only accurate if the norm is no
%             smaller than 1e-8 in size. A warning message will be printed
%             in those cases the computation is dubious.
%
% REFERENCES:
%   [1]         Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor 
%               Toolbox Version 2.6, Available online, February 2015. URL: 
%               http://www.sandia.gov/~tgkolda/TensorToolbox/.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     May 24, 2018

no_decimals = 500;

AA = cell(size(A));
for k = 1:length(A)
    AA{k} = A{k}.';
end

nX = norm(X);
nG = norm(G);
ip = SptTtInnerProductC(int32(X.subs.'), X.vals, G.data, AA);
computation = nX^2 + nG^2 - 2*ip;
if computation < 0
    fprintf('WARNING: The norm squared computation is taking on a negative value.\n');
    fprintf('\tThe computed norm is probably not accurate.\n');
    computation = abs(computation);
elseif sqrt(computation) < 1e-8
    fprintf('WARNING: The norm is smaller than 1e-8.\n');
    fprintf('\tThe computed norm is may not be accurate.\n');
end

nrm = sqrt(computation);

end