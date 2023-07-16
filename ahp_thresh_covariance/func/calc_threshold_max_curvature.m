function [theta,tspk] = calc_threshold_max_curvature(V,dim,nPoints,findpeaksArgs)
    if (nargin < 2) || isempty(dim)
        dim = [];
    end
    if (nargin < 3) || isempty(nPoints)
        nPoints = 15; % consider 15 points before the threshold
    end
    if (nargin < 4) || isempty(findpeaksArgs)
        findpeaksArgs = {'MinPeakProminence',30}; % peak is at least 30 mV larger than it's surroundings
    end

    if iscell(V)
        theta = cell(size(V));
        tspk = cell(size(V));
        for i = 1:numel(V)
            if ~isempty(V{i})
                [th,tt] = calc_threshold_max_curvature_internal(V{i},dim,nPoints,findpeaksArgs);
                th = matCell2Mat(th)';
                tt = matCell2Mat(tt)';
                if isempty(th)
                    theta{i} = NaN(size(V{i},2),1);
                    tspk{i} = NaN(size(theta{i}));
                else
                    theta{i} = matCell2Mat(th)';
                    tspk{i} = matCell2Mat(tt)';
                end
            end
        end
    else
        [theta,tspk] = calc_threshold_max_curvature_internal(V,dim,nPoints,findpeaksArgs);
    end
end

function [theta,tspk] = calc_threshold_max_curvature_internal(V,dim,nPoints,findpeaksArgs)
    sz = size(V);
    if any(sz==1) % if there is a singleton dimension (i.e. V is a row vector or col vector)
        dim = find(sz==1); % then we need dim to be equal to the singleton dimension (2 if a col vector, 1 if row vector)
    else % otherwise, the default is to run for each column of V
        if isempty(dim)
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
    n = size(V,2);
    if n > 1
        theta = cell(1,n);
        tspk = cell(1,n);
        for j = 1:n
            [theta{j},tspk{j}] = calc_thresh_vec(V(:,j),nPoints,findpeaksArgs);
        end
    else
        [theta,tspk] = calc_thresh_vec(V,nPoints,findpeaksArgs);
    end
end

function [theta,tspk] = calc_thresh_vec(V,nPoints,findpeaksArgs)
    [~,k] = findpeaks(V,findpeaksArgs{:});
    n = numel(k);
    theta = zeros(1,n);
    tspk = zeros(1,n);
    for i = 1:n
        % method of the maximum curvature, pg 46 Vinicius Lima MSc dissertation:
        % get the V that maximizes the function Kp = (d2V/dt2) * ( 1 + (dV/dt)**2 )**(-3/2)
        start = k(i)-nPoints;
        if start < 1
            start = 1;
        end
        Vspike = V(start:k(i));
        dV = diff(Vspike,1); % dV/dt
        d2V = diff(Vspike,2); % d2V/dt2
        Kp = d2V .* (1.0 + dV(1:(end-1)).^2).^(-1.5);
        [~,m] = max(Kp);
        m = m + 1;
        if isempty(m)
            m=1;
        end
        %fprintf('i=%d       ',i)
        %fprintf('m=%d\n',m)
        theta(i) = Vspike(m);
        tspk(i) = start-1 + m;
        %fprintf('theta = %g;    V(tspk) = %g\n',theta(i),V(tspk(i))); % debugging -> checking if tspk is correct
    end
    if size(V,2) == 1
        theta = theta(:);
        tspk = tspk(:);
    end
end