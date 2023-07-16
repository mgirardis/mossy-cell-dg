function [cf,un_f] = fit_by_feature(x,y,feat,fitArgs,skipError)
% feat is an array the size of x and y
% such that this function fits x(feat==feat(1)) vs. y(feat==feat(1)), etc
% for each unique feature
% if fitArgs is cell of cells, then each element of fitArgs will be used as the fitArgs for the fit of a given feature
%  [ thus it must contain one cell for each unique feature, sorted according to unique(feat) ]
% otherwise, it will be used as the fit args for all features
    if (nargin < 3) || isempty(feat)
        feat = [];
    end
    if (nargin < 4) || isempty(fitArgs)
        fitArgs = {'poly1'};
    end
    if (nargin < 5) || isempty(skipError)
        skipError = false;
    end
    if numel(x) ~= numel(y)
        error('x and y must have the same number of elements')
    end
    x = x(:);
    y = y(:);
    if isempty(feat)
        un_f = [];
        if isCellOfCells(fitArgs)
            fitArgs = fitArgs{1};
        end
        idx = ~(isnan(x) | isnan(y));
        cf = fit(x(idx),y(idx),fitArgs{:});
        return
    end
    feat = feat(:);
    if numel(x) ~= numel(feat)
        error('x,y and feat must have the same number of elements')
    end
    un_f = unique(feat);
    if ~isCellOfCells(fitArgs)
        if ~iscell(fitArgs)
            fitArgs = {fitArgs};
        end
        fitArgs = cellfun(@(x)fitArgs,cell(size(un_f)),'UniformOutput',false);
    end
    cf = cell(size(un_f));
    for i = 1:numel(un_f)
        f = un_f(i);
        try
            xx = x(feat==f);
            yy = y(feat==f);
            idx = ~(isnan(xx) | isnan(yy));
            cf{i} = fit(xx(idx),yy(idx),fitArgs{i}{:});
        catch e
            if strcmpi(e.identifier,'curvefit:fit:InsufficientData') || skipError
                cf{i} = [];
                if isnumeric(f)
                    f = num2str(f);
                end
                warning('fit_by_feature:fit:err','skipping feat = %s',f);
            else
                rethrow(e);
            end
        end
    end
end

function r = isCellOfCells(c)
    r = iscell(c);
    if r
        if isempty(c)
            r = false;
            return
        end
        r = r && all(cellfun(@iscell,c));
    end
end