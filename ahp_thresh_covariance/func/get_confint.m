function ci = get_confint(cf,varnum,level)
    if (nargin < 2) || isempty(varnum)
        varnum = 1;
    end
    if (nargin < 3) || isempty(level)
        level = 0.95;
    end
    try
        ci = confint(cf,level);
        ci = ci(:,varnum);
    catch
        ci = NaN(2,1);
    end
end