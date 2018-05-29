function [YsT, vecYs] = sketch_from_mat(J1, J2, h1int64, h2int64, s, verbose, filename, inc_size)
% sketch_from_mat   This sketch function returns the various sketches of a
%                   tensor stored in a mat file.
%
%                   This function requires Tensor Toolbox [1] version 2.6. 
%   
%   J1, J2 are the sketch dimensions; h1int64, h2int64, s are the various
%   hash functions; verbose is the verbose flag; filename is the path and
%   name of the mat file; inc_size is a vector that contains the increment 
%   size to be used when reading in the mat file tensor (must divide the 
%   corresponding tensor dimension size). 

file = matfile(filename, 'Writable', false);
sizeY = size(file, 'Y');
N = length(sizeY);
ns = 1:N;

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

end