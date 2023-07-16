clearvars
close all

addpath func

all_cells = load('..\2020-06-25\hMC_RM_noHoldingCurrent.mat');
dt = all_cells.Cell1_20170713R05_wSpike(2,1) - all_cells.Cell1_20170713R05_wSpike(1,1); % in ms

%% align by peak

[spks,theta,Vpeak,idx_threshold,idx_peak] = align_spikes(all_cells.Cell1_20170713R05_wSpike(:,2:end),2,20,12/dt,true,'peak');

figure;
plot(dt.*(1:size(spks,1)),spks,'-');
hold all;
plot(dt*idx_peak,Vpeak,'or','MarkerFaceColor','r');
plot(dt*idx_threshold,theta,'sb','MarkerFaceColor','b');
title('Align by peak');
ylabel('V(t) (mV)');
xlabel('t (ms)');

%% align by threshold

[spks,theta,Vpeak,idx_threshold,idx_peak] = align_spikes(all_cells.Cell1_20170713R05_wSpike(:,2:end),2,20,12/dt,true,'threshold');

figure;
plot(dt.*(1:size(spks,1)),spks,'-');
hold all;
plot(dt*idx_peak,Vpeak,'or','MarkerFaceColor','r');
plot(dt*idx_threshold,theta,'sb','MarkerFaceColor','b');
title('Align by threshold');
ylabel('V(t) (mV)');
xlabel('t (ms)');

%% plot V threshold vs. V reset

k = max(max(idx_peak)) + 9;

figure;
plot(theta,spks(k,:),'o')