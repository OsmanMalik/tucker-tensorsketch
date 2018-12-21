function [YsT, vecYs, vecYs_stop] = sketch_from_mat_ttmts(J1, J2, h1int64, h2int64, s, verbose, filename, inc_size)
% SKETCH_FROM_MAT_TTMTS     This sketch function returns the various
%                           sketches of a tensor stored in a mat file
%                           required for tucker_ttmts.
%
%                           This function requires Tensor Toolbox [1]
%                           version 2.6. 
%   
%   [YsT, vecYs, vecYs_stop] = SKETCH_FROM_MAT_TTMTS(J1, J2, h1int64, h2int64, s, verbose, filename, inc_size)
%   returns a cell YsT containing the sketches of each matricization of the
%   tensor stored in the mat file specified in the filename variable. The 
%   function also returns vecYs and vecYs_stop which are the sketches of
%   size J2 and J1, respectively, of the vectorization of that same tensor.
%   The variables J1 and J2 specify the target sketch dimensions; h1int64, 
%   h2int64, s are the various hash functions; verbose is the verbose flag 
%   (true or false); and inc_size is a vector that contains the increment 
%   size to be used when reading in the mat file tensor (must divide the 
%   corresponding tensor dimension size). 
%
%   Please see demo3.m, which was provided together with
%   this software, for an example of how to use this function.
%
%   For further information about our methods, please see our paper [2].
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

file = matfile(filename, 'Writable', false);
sizeY = size(file, 'Y');
N = length(sizeY);
ns = 1:N;
YsT = cell(N,1);

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
        vecYs_stop = zeros(J1, 1);
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
            vecYs_stop = vecYs_stop + TensorSketchVecC_git(Y_piece, h1int64, s, ...
                int64(J1), int64(slice_start), int64(slice_end));
        end
    end
    if verbose
        fprintf('Finished computing sketch %d out of %d...\n', n+1, N+1)
    end
end

end