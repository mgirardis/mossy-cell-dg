function lh = plot_by_feat(ax,x,y,feat,cmap,symbolList,plotArgs,setCMap)
% feat is an array the size of x and y
% such that this function plots x(feat==feat(1)) vs. y(feat==feat(1)), etc
% for each unique feature, in different colors (given by cmap) using different symbols
% from the symbolList
    figp = getDefaultFigureProperties();
    if isempty(ax)
        fig = figure;
        ax = axes;
    end
    if (nargin < 4) || isempty(feat)
        feat = [];
    end
    if (nargin < 5) || isempty(cmap)
        cmap = [];
    end
    if (nargin < 6) || isempty(symbolList)
        symbolList = figp.pSymbols;
    end
    if (nargin < 7) || isempty(plotArgs)
        plotArgs = {};
    end
    if (nargin < 8) || isempty(setCMap)
        setCMap = true;
    end
    get_val = @(x,v)x;
    get_symbol = @(s,k)s(mod(k,numel(s))+1);
    if ~isempty(feat)
        get_val = @(x,v)x(feat==v);
    end
    un_f = unique(feat);
    if isempty(cmap)
        if isempty(feat)
            cmap = [0,0,1];
        else
            cmap = jet(numel(un_f));%brewerCMap(numel(un_f),2);
        end
    end
    if isa(cmap,'function_handle')
        cmap = cmap(max(1,numel(un_f)));
    else
        n = max(1,numel(un_f));
        if size(cmap,1) ~= n
            cmap = interp1(1:size(cmap,1), cmap, linspace(1,size(cmap,1),n), 'linear');
        end
    end
    if isempty(feat)
        lh = plot(ax,x,y,get_symbol(symbolList,1),'Color',cmap(1,:),plotArgs{:});
    else
        lh = gobjects(1,numel(un_f));
        hold(ax,'on');
        for i = 1:numel(un_f)
            lh(i) = plot(ax,get_val(x,un_f(i)),get_val(y,un_f(i)),get_symbol(symbolList,i),'Color',cmap(i,:),'MarkerFaceColor',cmap(i,:),plotArgs{:});
        end
    end
    if setCMap
        colormap(ax,cmap);
    end
end