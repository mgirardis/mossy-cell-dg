function v = getPar(s,par)
    v = NaN;
    if ~isempty(s)
        v = s.(par);
    end
end