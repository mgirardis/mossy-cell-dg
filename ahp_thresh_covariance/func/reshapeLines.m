function B = reshapeLines(A,sz,outputType)
% outputType == '3darray' -> 3d array of matrices (default)
%               'cell' -> cell array of matrices
    if (nargin < 3) || isempty(outputType)
        outputType = '3darray';
    end
    n = size(A,1);
    if strcmpi(outputType,'3darray')
        B = zeros([sz(:)',n]);
        for i = 1:n
            B(:,:,i) = reshape(A(i,:),sz);
        end
    elseif strcmpi(outputType,'cell')
        B = cell(1,n);
        for i = 1:n
            B{i} = reshape(A(i,:),sz);
        end
    else
        error('Unknown outputType');
    end
end