function C = concat_all_cells(c,f)
    if (nargin < 2) || isempty(f)
        f = @(X)X;
    end
    C = reshape(cellfun(@(X)f(X),c,'UniformOutput',false),1,[]);
    m = max(cellfun(@(X)size(X,1),C));
    C = cell2mat(cellfun(@(X)fill_to_complete(X,[m,NaN]),C,'UniformOutput',false));
end