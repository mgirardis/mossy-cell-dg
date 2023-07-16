function [fitBest,tBest] = find_best_fit_V_vs_theta(V,theta,start_t)
% V -> rows is time; each col is a spike; all spikes (one per col) are aligned
% theta -> row vector of thresholds: each element j corresponds to the threshold of the spike in column j of V(:,j)
% start_t -> row of V to start the fitting process
%
% this function fits a line to the data of V(t,:) vs. theta for each t
% and returns the fit parameters (fitBest) and the t (tBest) of the best fit
% by minimizing the absolute mean residual of the fits
%
% according to
% Teeter et al (2018) Generalized leaky integrate-and-fire models classify multiple neuron types. Nat Comm 9:709.
% Fig 1A of the supp material
    if (nargin < 3) || isempty(start_t)
        start_t = 1;
    end
    theta = theta(:);
    T = size(V,1);
    res = inf(T,1);
    for t = start_t:T
        VV = reshape(V(t,:),[],1);
        idx = ~(isnan(VV) | isnan(theta));
        cf = fit(theta(idx),VV(idx),'poly1');
        res(t) = mean(abs(VV(idx) - cf(theta(idx)))); % mean absolute value of residuals of the fit
    end
    [~,tBest] = min(res);
    VV = reshape(V(tBest,:),[],1);
    idx = ~(isnan(VV) | isnan(theta));
    fitBest = fit(theta(idx),VV(idx),'poly1');
end