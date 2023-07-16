function [A,un_I] = join_cells_by_current(C,all_I,current_is_first_row_in_C)
    if (nargin < 3) || isempty(current_is_first_row_in_C)
        current_is_first_row_in_C = true;
    end
    if ~iscell(all_I)
        all_I = {all_I};
    end
    un_I = unique(cell2mat( cellfun(@(x)flatten(x),flatten(all_I),'UniformOutput',false) ));
    un_I = un_I(~isnan(un_I));
    m = max(cellfun(@(X)size(X,1),C));
    A = NaN(m,numel(un_I)+1); % first column is time
    dt = C{1}(end,1) - C{1}(end-1,1);
    n = zeros(1,numel(un_I)); % row index in which to insert the column of C corresponding to I
    for i = 1:numel(un_I)
        I = un_I(i); % for current I
        for j = 1:numel(C)
            if current_is_first_row_in_C
                k = find(C{j}(1,:) == I); % find the col in C{j} that corresponds to I
            else
                k = find(all_I{j} == I); % find the col in C{j} that corresponds to I
            end
            if ~isempty(k)
                if ~current_is_first_row_in_C
                    k = k + 1;
                end
                start_row = n(i) + 1;
                end_row = n(i) + numel(C{j}(2:end,k));
                A(start_row:end_row,i+1) = C{j}(2:end,k);
                n(i) = end_row;
            end
        end
    end
    A(:,1) = (1:size(A,1)).*dt;
end