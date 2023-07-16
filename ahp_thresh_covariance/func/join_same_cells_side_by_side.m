function a = join_same_cells_side_by_side(c)
    fn = fieldnames(c);
    cell_id = cellfun(@(x)str2double(x{1}(5:end)),regexp(fn,'^Cell\d+','match'));
    cell_id_un = unique(cell_id);
    for k = 1:numel(cell_id_un)
        r = cell_id_un(k);
        fields = fn(cell_id==r);
        [t_min,t_max,nCols] = get_min_max_time_cols(c,fields);
        time = join_time_col(c,fields);
        nT = numel(time);
        label = ['Cell',num2str(r)];
        a.(label) = NaN(nT+1,sum(nCols)+1);
        a.(label)(2:end,1) = time; % time column
        for i = 1:numel(fields)
            i1 = find(a.(label)(:,1) == t_min(i));
            i2 = find(a.(label)(:,1) == t_max(i));
            s = mycumsum(nCols(1:(i-1)));
            j1 = 2 + s(end);
            j2 = j1 + nCols(i) - 1;
            a.(label)(i1:i2,j1:j2) = c.(fields{i})(2:end,2:end);
            a.(label)(1,j1:j2) = c.(fields{i})(1,2:end);
        end
    end
end

function t = join_time_col(c,fn)
    t = c.(fn{1})(2:end,1);
    for i = 2:numel(fn)
        t = union(t,c.(fn{i})(2:end,1));
    end
end

function s = mycumsum(x)
    s = cumsum(x);
    if isempty(s)
        s = 0;
    end
end

function [t_min,t_max,nCols] = get_min_max_time_cols(c,fn)
    t_min = cellfun(@(x)min(c.(x)(:,1)),fn,'UniformOutput',true);
    t_max = cellfun(@(x)max(c.(x)(:,1)),fn,'UniformOutput',true);
    nCols = cellfun(@(x)size(c.(x),2)-1,fn,'UniformOutput',true);
    %n_t = structfun(@(x)size(c.(x),1)-1,fn,'UniformOutput',true);
    %dt = c.(fn{1})(3,1)-c.(fn{1})(2,1);
end