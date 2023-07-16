clearvars
close all

addpath func

all_cells = load('..\2020-06-25\hMC_StepCurrent_Data.mat');
dt = all_cells.Cell1_20180418R20(3,1) - all_cells.Cell1_20180418R20(2,1); % in ms

[all_cells,I,names] = join_same_cells(all_cells);
[all_cells_data, all_I] = join_cells_by_current(all_cells,I,false);

%% align spikes by threshold

[spks,theta,Vpeak,Vmin,idx_threshold,idx_peak,idx_min,current_labels,spk_num] = align_spikes(all_cells_data(:,2:end),2,int32(2/dt),int32(48/dt),true,'threshold',all_I);
spks(:,isnan(theta)) = [];
idx_threshold(isnan(theta))=[];
idx_peak(isnan(theta)) =[];
Vpeak(isnan(theta)) = [];
idx_min(isnan(theta)) =[];
Vmin(isnan(theta)) = [];
current_labels(isnan(theta)) = [];
spk_num(isnan(theta)) = [];
theta(isnan(theta))=[];

k = max(idx_peak);
[cf,tt] = find_best_fit_V_vs_theta(spks,theta,k);

[cf_curr,cf_curr_val]=fit_by_feature(theta,spks(tt,:),current_labels,'poly1');
[cf_spk,cf_spk_val]=fit_by_feature(theta,spks(tt,:),spk_num,'poly1');

%% plotting figure

figure;
% plot spikes
ax = subplot(2,2,1);
plot(ax,dt.*(1:size(spks,1)),spks,'-','HandleVisibility','off');
hold(ax,'all');
plot(ax,dt*idx_peak,Vpeak,'or','MarkerFaceColor','r','MarkerSize',3);
plot(ax,dt*idx_min,Vmin,'vk','MarkerFaceColor','k','MarkerSize',3);
plot(ax,dt*idx_threshold,theta,'sb','MarkerFaceColor','b','MarkerSize',3);
%title(ax,'Align by threshold');
ylabel(ax,'V(t) (mV)','FontSize',8);
xlabel(ax,'t (ms)','FontSize',8);
plot(ax,dt*tt*ones(size(theta)),spks(tt,:),'^m','MarkerFaceColor','m','MarkerSize',3);
legend(ax,{'AP peak','AHP min','Threshold','Estimated V_R (Teeter et al., 2018)'},'FontSize',8,'Location','northoutside');
xlim(ax,[0,dt*size(spks,1)]);

% plot V threshold vs. V reset (color currents)
%figure;
ax = subplot(2,2,2);
plot_by_feat(ax,theta,spks(tt,:),current_labels,[],'o',{'MarkerSize',3});
hold on
plot(theta,cf(theta),'-k','LineWidth',2);
xlabel('\theta, threshold (mV)','FontSize',8)
ylabel('Estimated V_R (mV), Teeter et al 2018','FontSize',8)
% title('Align by threshold');
set(ax,'CLim',minmax(current_labels(:)'),'Layer','top','Box','on');
cbar = colorbar(ax,'Position',[0.9 0.78 0.0085 0.1586]);
set(cbar.Label,'String','$I_{inj}$','Interpreter','latex','FontSize',14,'Rotation',0,'Units','normalized');
cbar.Label.Position = [2.8889 1.3072 0];

ax = axes('Position',[0.8175 0.63952380952381 0.145714285714283 0.137619047619051]);
V_theta_slope_curr = cellfun(@(f)f.p1,cf_curr);
plot(ax,cf_curr_val,V_theta_slope_curr,'-o','MarkerSize',4,'MarkerFaceColor','w','Color','b');
set(ax,'YScale','log');
xlabel(ax,'$I_{inj}$','Interpreter','latex','FontSize',10);
ylabel(ax,'Slope','Interpreter','latex','FontSize',10);

% plot V threshold vs. V reset (color spike #)
%figure;
ax = subplot(2,2,4);
plot_by_feat(ax,theta,spks(tt,:),spk_num,[],'o',{'MarkerSize',3});
hold on
plot(theta,cf(theta),'-k','LineWidth',2);
xlabel('\theta, threshold (mV)','FontSize',8)
ylabel('Estimated V_R (mV), Teeter et al 2018','FontSize',8)
% title('Align by threshold');
set(ax,'CLim',minmax(spk_num(:)'),'Layer','top','Box','on');
cbar = colorbar(ax,'Position',[0.9 0.3 0.0085 0.1586]);
set(cbar.Label,'String','Sp\#','Interpreter','latex','FontSize',14,'Rotation',0,'Units','normalized');
cbar.Label.Position = [2.8889 1.3072 0];

% plot V reset vs. AHP min (color spike #)
%figure;
ax = subplot(2,2,3);
plot_by_feat(ax,spks(tt,:),Vmin,current_labels,[],'o',{'MarkerSize',3});%plot_by_feat(ax,spks(tt,:),Vmin,spk_num,[],'o',{'MarkerSize',3});
xlabel('Estimated V_R (mV), Teeter et al 2018','FontSize',8)
ylabel('AHP min (mV)','FontSize',8)
% title('Align by threshold');
set(ax,'CLim',minmax(current_labels(:)'),'Layer','top','Box','on');
cbar = colorbar(ax,'Position',[0.42 0.3 0.0085 0.1586]);
set(cbar.Label,'String','$I_{inj}$','Interpreter','latex','FontSize',14,'Rotation',0,'Units','normalized');
cbar.Label.Position = [2.8889 1.3072 0];

%% plotting estimated VR vs theta with fit by feature (for each current and each spk# separately)

% plot V threshold vs. V reset (color currents)
figure;
ax = subplot(2,1,1);
plot_by_feat(ax,theta,spks(tt,:),current_labels,[],'o',{'MarkerSize',3});
hold on
cmap = jet(numel(cf_curr));
for i = 1:numel(cf_curr)
    if ~isempty(cf_curr{i})
        plot(theta,cf_curr{i}(theta),'-','LineWidth',0.5,'Color',cmap(i,:),'DisplayName',sprintf('I=%g',cf_curr_val(i)));
    end
end
xlabel(ax,'\theta, threshold (mV)','FontSize',8)
ylabel(ax,'Estimated V_R (mV), Teeter et al 2018','FontSize',8)
set(ax,'XLim',minmax(theta(:)'),'YLim',minmax(spks(tt,:)),'CLim',minmax(current_labels(:)'),'Layer','top','Box','on');
cbar = colorbar(ax,'Position',[0.9 0.78 0.0085 0.1586]);
set(cbar.Label,'String','$I_{inj}$','Interpreter','latex','FontSize',14,'Rotation',0,'Units','normalized');
cbar.Label.Position = [2.8889 1.3072 0];

ax = subplot(2,1,2);
ax_pos = ax.Position;
V_theta_slope_curr = cellfun(@(f)f.p1,cf_curr);
valid_ind = V_theta_slope_curr>0;
plot(ax,cf_curr_val(valid_ind),V_theta_slope_curr(valid_ind),'-o','MarkerSize',4,'MarkerFaceColor','w','Color','b');
set(ax,'YScale','log');
xlabel(ax,'$I_{inj}$','Interpreter','latex','FontSize',10);
ylabel(ax,'Slope','Interpreter','latex','FontSize',10);
ax.Position(3) = (ax.Position(3)/2) - ax.Position(1)/2;
plotHorizontalLines(ax,0.8,'LineStyle','--','Color','k');

ax = axes('Position',ax_pos);
V_theta_disp_curr = cellfun(@(f)f.p2,cf_curr);
plot(ax,cf_curr_val(valid_ind),V_theta_disp_curr(valid_ind),'-o','MarkerSize',4,'MarkerFaceColor','w','Color','b');
% set(ax,'YScale','log');
xlabel(ax,'$I_{inj}$','Interpreter','latex','FontSize',10);
ylabel(ax,'Displacement (mV)','Interpreter','latex','FontSize',10);
ax.Position(3) = (ax.Position(3)/2) - ax.Position(1)/2;
ax.Position(1) = ax.Position(1)*2 + ax.Position(3);
plotHorizontalLines(ax,-12,'LineStyle','--','Color','k');

%%

% plot V threshold vs. V reset (color spike #)
figure;
ax = axes;
plot_by_feat(ax,theta,spks(tt,:),spk_num,[],'o',{'MarkerSize',3});
hold on
cmap = jet(numel(cf_spk));
for i = 1:numel(cf_spk)
    if ~isempty(cf_spk{i})
        plot(theta,cf_spk{i}(theta),'-','LineWidth',0.5,'Color',cmap(i,:),'DisplayName',sprintf('sp \# %g',cf_spk_val(i)));
    end
end
xlabel('\theta, threshold (mV)','FontSize',8)
ylabel('Estimated V_R (mV), Teeter et al 2018','FontSize',8)
% title('Align by threshold');
set(ax,'CLim',minmax(spk_num(:)'),'Layer','top','Box','on');
cbar = colorbar(ax,'Position',[0.9 0.3 0.0085 0.1586]);
set(cbar.Label,'String','Sp\#','Interpreter','latex','FontSize',14,'Rotation',0,'Units','normalized');
cbar.Label.Position = [2.8889 1.3072 0];


%% plotting vs. AHP min

% plot V threshold vs. AHP min (color currents)
figure;
ax = axes;
plot_by_feat(ax,theta,Vmin,current_labels,[],'o',{'MarkerSize',3});
hold on
plot(theta,cf(theta),'-k','LineWidth',2);
xlabel('\theta, threshold (mV)')
ylabel('AHP min (mV)')
title('Align by threshold');
set(ax,'CLim',minmax(current_labels(:)'),'Layer','top','Box','on');
cbar = colorbar(ax,'Position',[0.8376 0.7024 0.0085 0.1586]);
set(cbar.Label,'String','$I_{inj}$','Interpreter','latex','FontSize',14,'Rotation',0,'Units','normalized');
cbar.Label.Position = [2.8889 1.3072 0];

% plot V threshold vs. AHP min (color spike #)
figure;
ax = axes;
plot_by_feat(ax,theta,Vmin,spk_num,[],'o',{'MarkerSize',3});
hold on
plot(theta,cf(theta),'-k','LineWidth',2);
xlabel('\theta, threshold (mV)')
ylabel('AHP min (mV)')
title('Align by threshold');
set(ax,'CLim',minmax(spk_num(:)'),'Layer','top','Box','on');
cbar = colorbar(ax,'Position',[0.8376 0.7024 0.0085 0.1586]);
set(cbar.Label,'String','Spike \#','Interpreter','latex','FontSize',14,'Rotation',0,'Units','normalized');
cbar.Label.Position = [2.8889 1.3072 0];