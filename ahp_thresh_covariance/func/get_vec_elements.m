function el = get_vec_elements(v,idx)
    c1 = idx < 1;
    c2 = idx > numel(v);
    k = ~c1 & ~c2;
    %try
    el(k) = v(idx(k));
    %catch
    %    disp('error')
    %end
    el(c1)=NaN;
    el(c2)=NaN;
end