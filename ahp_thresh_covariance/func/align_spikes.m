function [spks,theta,Vpeak,Vmin,idx_threshold,idx_peak,idx_ahpmin,labels,spk_num] = align_spikes(V,dim,nRewind,nForward,align_all,align_location,dim_labels,nPoints,findpeaksArgs,spkFeatAsMatrix)
% aligns every spike found in D
% dim -> 1 or 2;
%        if 1, each line of D is treated as an independent spike train, if 2, then every column in D is a spike train
% nRewind -> number of time steps considered before the peak
% nForward -> number of time steps considered after the peak
% align_all -> a bool flag; if true, then all spikes in D are aligned; otherwise only align spikes within their own spike trains
% align_location -> 'peak' or 'threshold'; aligns spikes either by the time of their peak (max height)
%                                          or by the time of their threshold (detected by max curvature)
% dim_labels -> label for each column or row of V (column if dim == 2, row if dim == 1)
%               it is returned as a label for each column in spikes
% nPoints -> consider nPoints before the peak to calculate the threshold by max curvature
% spkFeatAsMatrix -> if true, returns spike features as a matrix (each col corresponds to a col in V, or row in V if dim = 1)
%                    if false, returns a row vector, where each element corresponds to a col in spks return variable
%                    this parameter is only useful if align_all=true
% findPeaksArgs -> arguments sent to findpeaks function to detect spikes
    sz = size(V);
    if any(sz==1) % if there is a singleton dimension (i.e. V is a row vector or col vector)
        dim = find(sz==1); % then we need dim to be equal to the singleton dimension (2 if a col vector, 1 if row vector)
    else % otherwise, the default is to run for each column of V
        if (nargin < 2) || isempty(dim)
            dim = 2; % dim = 1: finds peaks for each line; dim = 2 finds peaks for each column of V
        end
    end
    if (dim == 1)
        if isvector(V)
            V = V(:);
        else
            V = V';
        end
    end
    if (nargin < 3) || isempty(nRewind)
        nRewind = 20;
    end
    if (nargin < 4) || isempty(nForward)
        nForward = 50;
    end
    if (nargin < 5) || isempty(align_all)
        align_all = true;
    end
    if (nargin < 6) || isempty(align_location)
        align_location = 'peak'; % 'peak' or 'threshold'
    end
    if (nargin < 7) || isempty(dim_labels)
        dim_labels = {};
    end
    if (nargin < 8) || isempty(nPoints)
        nPoints = 20;
    end
    if (nargin < 9) || isempty(findpeaksArgs)
        findpeaksArgs = {'MinPeakProminence',30}; % peak is at least 30 mV larger than it's surroundings
    end
    if (nargin < 10) || isempty(spkFeatAsMatrix)
        spkFeatAsMatrix = false;
    end
    if strcmpi(align_location,'peak')
        [~,idx] = findpeaks_in_cols(V,findpeaksArgs);
    elseif strcmpi(align_location,'threshold')
        [~,idx] = calc_threshold_max_curvature(V,dim,nPoints,findpeaksArgs);
    else
        error('unknown align location');
    end
    if ~iscell(idx)
        idx = {idx};
    end
    n = numel(idx);
    spks = cell(1,numel(idx));
    for i = 1:n
        if isempty(idx{i})
            continue
        end
        spks{i} = get_V_around_idx(V(:,i),idx{i},nRewind,nForward,true);
    end
    [Vpeak,idx_peak] = findpeaks_in_cols(spks,[findpeaksArgs,'NPeaks',1]);
    [theta,idx_threshold] = calc_threshold_max_curvature(spks,dim,nPoints,[findpeaksArgs,'NPeaks',1]);
    [Vmin,idx_ahpmin]= find_min_after_peaks(spks,idx_peak);
    spk_num = cellfun(@(x)1:numel(x),idx_peak,'UniformOutput',false);
    if align_all
        spks = cell2mat(spks);
        if spkFeatAsMatrix
            labels = matCell2Mat(get_labels_as_matrix(dim_labels,theta))';
            theta = matCell2Mat(theta)';
            idx_threshold = matCell2Mat(idx_threshold)';
            idx_peak = matCell2Mat(idx_peak)';
            Vpeak = matCell2Mat(Vpeak)';
            idx_ahpmin = matCell2Mat(idx_ahpmin)';
            Vmin = matCell2Mat(Vmin)';
            spk_num = matCell2Mat(spk_num)';
        else
            reshapeVar = @(v)cellfun(@(x)reshape(x,1,[]),reshape(v,1,[]),'UniformOutput',false);
            if isempty(dim_labels)
                labels = [];
            else
                labels = repeatElements_local(dim_labels,cellfun(@numel,theta));
            end
            theta = cell2mat(reshapeVar(theta));
            idx_threshold = cell2mat(reshapeVar(idx_threshold));
            idx_peak = cell2mat(reshapeVar(idx_peak));
            Vpeak = cell2mat(reshapeVar(Vpeak));
            idx_ahpmin = cell2mat(reshapeVar(idx_ahpmin));
            Vmin = cell2mat(reshapeVar(Vmin));
            spk_num = cell2mat(reshapeVar(spk_num));
        end
    else
        labels = dim_labels;
        if numel(spks) == 1
            spks = spks{1};
        end
        if iscell(spk_num) && numel(spk_num) == 1
            spk_num = spk_num{1};
        end
        if iscell(idx_ahpmin) && numel(idx_ahpmin) == 1
            idx_ahpmin = idx_ahpmin{1};
        end
        if iscell(Vmin) && numel(Vmin) == 1
            Vmin = Vmin{1};
        end
        if iscell(Vpeak) && numel(Vpeak) == 1
            Vpeak = Vpeak{1};
        end
        if iscell(idx_threshold) && numel(idx_threshold) == 1
            idx_threshold = idx_threshold{1};
        end
        if iscell(idx_peak) && numel(idx_peak) == 1
            idx_peak = idx_peak{1};
        end
        if iscell(theta) && numel(theta) == 1
            theta = theta{1};
        end
    end
end

function l = get_labels_as_matrix(dim_labels,theta_cell)
    if ~iscell(dim_labels)
        l = num2cell(dim_labels);
    end
    l = cellfun(@(x,n)repeatElements_local(x,n),l,num2cell(cellfun(@numel,theta_cell)),'UniformOutput',false);
end

function [Vmin,idx] = find_min_after_peaks(V,idx_peak)
    if iscell(V)
        Vmin = cell(size(V));
        idx = cell(size(V));
        for i = 1:numel(V)
            [VV,kk] = find_min_after_peaks(V{i},idx_peak{i});
            if ~isempty(VV)
                Vmin{i} = matCell2Mat(VV)';
                idx{i} = matCell2Mat(kk)';
            end
        end
    else
        %if isempty(idx_peak)
        %    Vmin = NaN;
        %    idx = NaN;
        %    return
        %end
        Vmin = zeros(1,size(V,2));
        idx = zeros(size(Vmin));
        for j = 1:size(V,2)
            k = idx_peak(j);
            if isnan(k)
                Vmin(j) = NaN;
                idx(j) = NaN;
            else
                V(1:k,j) = NaN;
                [VV,kk] = mink(V(:,j),10); % finds 10 minimum in V(:,j)
                [idx(j),m] = min(kk); % chooses the minimum that is closer to the spike (i.e. least index)
                Vmin(j) = VV(m);
            end
        end
    end
end

function [Vpeak,idx] = findpeaks_in_cols(V,args)
    if iscell(V)
        Vpeak = cell(size(V));
        idx = cell(size(V));
        for i = 1:numel(V)
            [VV,kk] = findpeaks_in_cols(V{i},args);
            VV = matCell2Mat(VV)';
            kk = matCell2Mat(kk)';
            if isempty(VV)
                Vpeak{i} = NaN(size(V{i},2),1);
                idx{i} = NaN(size(Vpeak{i}));
            else
                Vpeak{i} = matCell2Mat(VV)';
                idx{i} = matCell2Mat(kk)';
            end
        end
    else
        N = size(V,2);
        Vpeak = cell(1,N);
        idx = cell(1,N);
        for i = 1:N
            [Vpeak{i},idx{i}] = findpeaks(V(:,i),args{:});
        end
    end
end

function xx = repeatElements_local(x,n)
% n = vector same size as x
% repeats n(i) times each element x(i) and returns it in the same order as in x
    N = sum(n); % total number of elements
    ind = zeros(1,N);
    ind(1:n(1)) = ones(1,n(1));
    s = n(1);
    for k = 2:numel(n)
        ind((s+1):(s+n(k))) = k.*ones(1,n(k));
        s = s + n(k);
    end
    xx = x(ind);
end