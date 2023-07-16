clearvars
close all

all_cells = load('..\2020-06-25\hMC_RM_noHoldingCurrent.mat');

data_label1 = 'Cell1_20170713R05_wSpike';
data_label2 = 'Cell1_20170713R06_wSpike';
data_label3 = 'Cell1_20170713R07_wSpike';

%%

get_el = @(x,k) x(1:k);

V = reshape(all_cells.(data_label1)(:,2:end),[],1);
V = [reshape(all_cells.(data_label1)(:,2:end),[],1),...
     reshape(all_cells.(data_label2)(:,2:end),[],1),...
     get_el(reshape(all_cells.(data_label3)(:,2:end),[],1),numel(V))]; % col -> realizations, rows -> time
t = (all_cells.(data_label1)(2,1) - all_cells.(data_label1)(1,1)) .* (1:size(V,1));

fh = figure('Position',[345 320 1000 100]);
ax = axes('Position',[0.045,0.25,0.945,0.7]);
plot(ax,t,V);
ax.XLim = minmax(t(:)');
ax.Box = 'off';
ylabel(ax,'V (mV)');
ax.XTickLabel = ax.XTick / 1e3;
xlabel(ax,'Time (s)');
ax.FontSize = 10;
ax.XLabel.Position(2) = ax.XLabel.Position(2)+20;
ax.YLabel.Position(1) = ax.YLabel.Position(1)+0.01e4;