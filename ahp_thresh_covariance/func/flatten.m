function a = flatten(a,asCol)
    if (nargin < 2) || isempty(asCol)
        asCol = false;
    end
    if asCol
        f = @(x)reshape(x,[],1);
    else
        f = @(x)reshape(x,1,[]);
    end
    a = f(a);
end