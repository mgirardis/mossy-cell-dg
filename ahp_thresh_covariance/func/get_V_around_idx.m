function VV = get_V_around_idx(V,idx,nRewind,nForward,returnAsMatrix)
    if (nargin < 3) || isempty(nRewind)
        nRewind = 20;
    end
    if (nargin < 4) || isempty(nForward)
        nForward = 50;
    end
    if (nargin < 5) || isempty(returnAsMatrix)
        returnAsMatrix = true;
    end

    VV = cell(size(idx));
    for i = 1:numel(idx)
        VV{i} = get_vec_elements(V,(idx(i) - nRewind):(idx(i) + nForward));
    end
    if returnAsMatrix
        VV = matCell2Mat(VV)';
    end
end