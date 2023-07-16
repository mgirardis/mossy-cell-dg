function [c,un_I,names] = join_same_cells(s)
% s -> struct with fields named 'Cell%d_...'
% assuming first line of each field in s is the input current and the first col is time:
% s.(field)(1,2:end) -> input current
% s.(field)(2:end,1) -> time
%
% this function joins the columns (one on top of the other) corresponding to the same cell and the same current
% and returns a cell array in which each element is a matrix representing a different cell in the original struct
% and each col in this matrix is a different current for that cell
    fn = fieldnames(s);
    names = get_cell_name(fn);
    celln = get_cell_number(fn);
    I = get_input_currents(s);
    [un_celln,un_id] = unique(celln);
    names = names(un_id);
    c = cell(1,numel(un_celln));
    un_I = cell(size(c));
    for k = 1:numel(un_celln)
        n = un_celln(k);
        [c{k},un_I{k}] = join_cells_by_current(struct2cell(get_substruct(s,fn(celln==n))),I(celln==n));
    end
end

function I = get_input_currents(s)
    I = struct2cell(structfun(@(X)X(1,:),s,'UniformOutput',false));
end

function n = get_cell_number(fn)
    n = cellfun(@(n)str2double(n.celln),regexp(fn,'Cell(?<celln>\d+)_','names'));
end

function n = get_cell_name(fn)
    n = cellfun(@(n)n.name,regexp(fn,'^(?<name>Cell\d+)_','names'),'UniformOutput',false);
end

function s = get_substruct(s,fields_to_keep)
    s = rmfield(s,setdiff(fieldnames(s),fields_to_keep));
end