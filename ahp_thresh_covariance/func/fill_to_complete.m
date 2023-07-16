function B = fill_to_complete(A,sz,v)
    if (nargin < 3) || isempty(v)
        v = NaN;
    end
    sA = size(A);
    sB = [max(sA(1),sz(1)),max(sA(2),sz(2))];
    B = v.*ones(sB);
    B(1:sA(1),1:sA(2)) = A;
end