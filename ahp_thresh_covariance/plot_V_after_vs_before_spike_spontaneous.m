clearvars
close all

addpath func

all_cells = load('..\2020-06-25\hMC_RM_noHoldingCurrent.mat');
dt = all_cells.Cell1_20170713R05_wSpike(2,1) - all_cells.Cell1_20170713R05_wSpike(1,1); % in ms

all_cells_data = concat_all_cells(struct2cell(all_cells),@(X)X(:,2:end));

%% align by peak
[spks,theta,Vpeak,Vmin,idx_threshold,idx_peak,idx_min,current_labels,spk_num] = align_spikes(all_cells_data,2,int32(4/dt),int32(14/dt),true,'peak');

% plot spikes
figure;
ax = axes;
plot(ax,dt.*(1:size(spks,1)),spks,'-');
hold(ax,'all');
plot(ax,dt*idx_peak,Vpeak,'or','MarkerFaceColor','r');
plot(ax,dt*idx_min,Vmin,'vk','MarkerFaceColor','k');
plot(ax,dt*idx_threshold,theta,'sb','MarkerFaceColor','b');
title(ax,'Align by peak');
ylabel(ax,'V(t) (mV)');
xlabel(ax,'t (ms)');

k = max(idx_peak);
[cf,tt] = find_best_fit_V_vs_theta(spks,theta,k);
plot(ax,dt*tt*ones(size(theta)),spks(tt,:),'^m','MarkerFaceColor','m');

% plot V threshold vs. V reset
figure;
plot(theta,spks(tt,:),'ob','MarkerFaceColor','b','MarkerSize',3);
hold on
plot(theta,Vmin,'sk','MarkerFaceColor','k','MarkerSize',3);
plot(theta,cf(theta),'-r');
xlabel('\theta, threshold (mV)')
ylabel('V_R (mV)')
title('Align by peak');

%% align by threshold
[spks,theta,Vpeak,Vmin,idx_threshold,idx_peak,idx_min,current_labels,spk_num] = align_spikes(all_cells_data,2,int32(2/dt),int32(14/dt),true,'threshold');

% plot spikes
figure;
ax = axes;
plot(ax,dt.*(1:size(spks,1)),spks,'-');
hold(ax,'all');
plot(ax,dt*idx_peak,Vpeak,'or','MarkerFaceColor','r');
plot(ax,dt*idx_min,Vmin,'vk','MarkerFaceColor','k');
plot(ax,dt*idx_threshold,theta,'sb','MarkerFaceColor','b');
title(ax,'Align by threshold');
ylabel(ax,'V(t) (mV)');
xlabel(ax,'t (ms)');

k = max(idx_peak);
[cf,tt] = find_best_fit_V_vs_theta(spks,theta,k);
plot(ax,dt*tt*ones(size(theta)),spks(tt,:),'^m','MarkerFaceColor','m');

% plot V threshold vs. V reset
figure;
plot(theta,spks(tt,:),'ob','MarkerFaceColor','b','MarkerSize',3);
hold on
plot(theta,Vmin,'sk','MarkerFaceColor','k','MarkerSize',3);
plot(theta,cf(theta),'-r');
xlabel('\theta, threshold (mV)')
ylabel('V_R (mV)')
title('Align by threshold');