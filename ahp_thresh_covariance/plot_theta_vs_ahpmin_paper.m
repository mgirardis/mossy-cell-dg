clearvars
close all

set(0,'defaultAxesFontName','CMU Sans Serif');
set(0,'defaultTextFontName','CMU Sans Serif');

addpath func

saveOutputFigures = false;

load('colors_for_paper.mat');
color_mc = color_mc(:,1:3);
color_mod = color_mod(:,1:3);
%color_mc = flipud(color_mc(:,1:3));
%color_mod = flipud(color_mod(:,1:3));

% mc_all_cells = load('..\2020-06-25\hMC_StepCurrent_DataJune2021.mat');
mc_all_cells = load('..\2020-06-25\hMC_StepCurrent_DataSept2021.mat');
dt_exp = mc_all_cells.Cell1_20180418R20(3,1) - mc_all_cells.Cell1_20180418R20(2,1); % in ms
[mc_all_cells,I_mc,names_mc] = join_same_cells(mc_all_cells);
[mc_all_cells_data, mc_all_I] = join_cells_by_current(mc_all_cells,I_mc,false);


%mod_all_cells = load('D:/Dropbox/p/uottawa/data/mossy_cell_experiment/model_data_paper/with_K_curr/currinj_spk_feat_EIFDTBoundSigKLR_volt_1.mat');%
%out_fname_suffix='withKcurr';
mod_all_cells = load('D:/Dropbox/p/uottawa/data/mossy_cell_experiment/model_data_paper/no_K_curr/currinj_spk_feat_EIFDTBoundSigKLR_volt_1.mat');
out_fname_suffix='noKcurr';

dt_mod = mod_all_cells.Cell1_EIFDTBoundSigKLR_R1(3,1) - mod_all_cells.Cell1_EIFDTBoundSigKLR_R1(2,1); % in ms
[mod_all_cells,I_mod,names_mod] = join_same_cells(mod_all_cells);
[mod_all_cells_data, mod_all_I] = join_cells_by_current(mod_all_cells,I_mod,false);

%% align spikes by threshold (mossy cells)
%                                                                                                          align_spikes(                         V,dim,        nRewind,        nForward,align_all,align_location,dim_labels,nPoints,findpeaksArgs,spkFeatAsMatrix)
[spks_mc,theta_mc,Vpeak_mc,Vmin_mc,idx_threshold_mc,idx_peak_mc,idx_min_mc,current_labels_mc,spk_num_mc] = align_spikes(mc_all_cells_data(:,2:end),  2,int32(2/dt_exp),int32(48/dt_exp),     true,   'threshold',  mc_all_I,     [], {'MinPeakProminence',30});
spks_mc(:,isnan(theta_mc)) = [];
idx_threshold_mc(isnan(theta_mc))=[];
idx_peak_mc(isnan(theta_mc)) =[];
Vpeak_mc(isnan(theta_mc)) = [];
idx_min_mc(isnan(theta_mc)) =[];
Vmin_mc(isnan(theta_mc)) = [];
current_labels_mc(isnan(theta_mc)) = [];
spk_num_mc(isnan(theta_mc)) = [];
theta_mc(isnan(theta_mc))=[];

k = max(idx_peak_mc);
[cf_mc,tt_mc] = find_best_fit_V_vs_theta(spks_mc,theta_mc,k);

[cf_curr_mc,cf_curr_val_mc]=fit_by_feature(theta_mc,spks_mc(tt_mc,:),current_labels_mc,'poly1');
[cf_spk_mc,cf_spk_val_mc]=fit_by_feature(theta_mc,spks_mc(tt_mc,:),spk_num_mc,'poly1');

%% align spikes by threshold (model)

[spks_mod,theta_mod,Vpeak_mod,Vmin_mod,idx_threshold_mod,idx_peak_mod,idx_min_mod,current_labels_mod,spk_num_mod] = align_spikes(mod_all_cells_data(:,2:end),2,int32(2/dt_mod),int32(8/dt_mod),true,'threshold',mod_all_I);
spk_idx = find(dt_mod.*idx_min_mod>6); % remove spikes with min after 6*dt ms
spks_mod(:,spk_idx) = [];
idx_threshold_mod(spk_idx)=[];
idx_peak_mod(spk_idx) =[];
Vpeak_mod(spk_idx) = [];
idx_min_mod(spk_idx) =[];
Vmin_mod(spk_idx) = [];
current_labels_mod(spk_idx) = [];
spk_num_mod(spk_idx) = [];
theta_mod(spk_idx)=[];

k = max(idx_peak_mod)+1;
[cf_mod,tt_mod] = find_best_fit_V_vs_theta(spks_mod,theta_mod,k);

[cf_curr_mod,cf_curr_val_mod]=fit_by_feature(theta_mod,spks_mod(tt_mod,:),current_labels_mod,'poly1');
[cf_spk_mod,cf_spk_val_mod]=fit_by_feature(theta_mod,spks_mod(tt_mod,:),spk_num_mod,'poly1');

cf_curr_val_mod = cf_curr_val_mod .* 1e3; % nA to pA

%% plotting figure

fill_between_Y = @(X,Y_top,Y_bottom,color) fill([X', fliplr(X')], [Y_top,fliplr(Y_bottom)], color,'HandleVisibility','off');

confint_slope_mc = cell2mat(cellfun(@(x)get_confint(x,1),cf_curr_mc,'UniformOutput',false)');
confint_slope_mc(1,confint_slope_mc(1,:)<0) = 1e-2; % I'll plot in logscale, so no value can be negative
confint_slope_mod = cell2mat(cellfun(@(x)get_confint(x,1),cf_curr_mod,'UniformOutput',false)');
confint_slope_mod(1,confint_slope_mod(1,:)<0) = 1e-2; % I'll plot in logscale, so no value can be negative

% plot V threshold vs. AHP min (color currents)
fh = figure('Position', [488 478.6000 441.8000 283.4000]);
ax1 = axes('FontName','Arial','FontSize',9,'Box','off','Position',[0.0956 0.1411 0.4589 0.8064]);%[0.1264 0.1481 0.7750 0.7966]);
plot_by_feat(ax1,theta_mc,Vmin_mc,current_labels_mc,color_mc,'o',{'MarkerSize',3});
plot_by_feat(ax1,theta_mod,Vmin_mod,current_labels_mod,color_mod,'s',{'MarkerSize',3},false);
hold on
%plot(theta_mc,cf_mc(theta_mc),'-k','LineWidth',2);
xlabel('Spike Threshold (mV)','FontSize',12)
ylabel('AHP minimum (mV)','FontSize',12)
set(ax1,'CLim',minmax(current_labels_mc(:)'),'Layer','top');
cbar1 = colorbar(ax1,'Location','west');
set(cbar1.Label,'String',{'\textbf{\textsf{hMC}}','\textbf{\textsf{$I_{\rm ext}$ (pA)}}'},'Interpreter','latex','FontSize',10,'Rotation',0,'Units','normalized');
cbar1.Position =  [0.1364 0.7559 0.0119 0.1327];
drawnow
cbar1.Label.Position = [1.6786 1.7872 0];%[2.1389 1.3981 0];
ax1.XLim = [-57 -15];

ax2 = axes('Position', [0.5958 0.5045 0.3766 0.4178]);%[0.6071 0.2399 0.3766 0.4178]);%[0.6270 0.2709 0.3133 0.2654]);
V_theta_slope_curr_mc = cellfun(@(f)getPar(f,'p1'),cf_curr_mc);
V_theta_slope_curr_mod = cellfun(@(f)getPar(f,'p1'),cf_curr_mod);
%V_theta_slope_curr_mc = cellfun(@(f)f.p1,cf_curr_mc);
%V_theta_slope_curr_mod = cellfun(@(f)f.p1,cf_curr_mod);

% I will plot in logscale, so only postive values allowed for the slope
confint_slope_mc_pos = confint_slope_mc(:,V_theta_slope_curr_mc>0);
cf_curr_val_mc_pos = cf_curr_val_mc(V_theta_slope_curr_mc>0);
V_theta_slope_curr_mc_pos = V_theta_slope_curr_mc(V_theta_slope_curr_mc>0);
confint_slope_mod_pos = confint_slope_mod(:,V_theta_slope_curr_mod>0);
cf_curr_val_mod_pos = cf_curr_val_mod(V_theta_slope_curr_mod>0);
V_theta_slope_curr_mod_pos = V_theta_slope_curr_mod(V_theta_slope_curr_mod>0);

plotHorizontalLines(ax2,0.8,'LineStyle','--','Color','k','LineWidth',0.1,'HandleVisibility','off','XMin',0,'XMax',400);
hold(ax2,'on');
plot(ax2,cf_curr_val_mc_pos,V_theta_slope_curr_mc_pos,'o:','MarkerSize',4,'Color',color_mc(13,:),'MarkerFaceColor','w','DisplayName','hMC');
ph = fill_between_Y(cf_curr_val_mc_pos,confint_slope_mc_pos(2,:),confint_slope_mc_pos(1,:),color_mc(13,:));
set(ph, 'edgecolor', 'none');
set(ph, 'FaceAlpha', 0.2);
plot(ax2,cf_curr_val_mod_pos,V_theta_slope_curr_mod_pos,'s:','MarkerSize',5,'Color',color_mod(13,:),'MarkerFaceColor','w','DisplayName','Model');
ph = fill_between_Y(cf_curr_val_mod_pos,confint_slope_mod_pos(2,:),confint_slope_mod_pos(1,:),color_mod(13,:));
set(ph, 'edgecolor', 'none');
set(ph, 'FaceAlpha', 0.2);
%set(ax2,'YScale','log');
xlabel(ax2,'\textbf{\textsf{Injected current, $I_{\rm ext}$ (pA)}}','FontName','CMU Sans Serif','Interpreter','latex','FontSize',12);
%ylabel(ax2,'Slope','FontSize',10);
title(ax2,'Fitted mean slope','FontSize',12,'FontWeight','bold');
ax2.FontName       = 'CMU Sans Serif';
%ax2.YLim              = [1e-4,2];
%ax2.YTick             = [0.01,0.8];
ax2.YTick(ax2.YTick==1)=0.8;
ax2.FontSize          = 9;
ax2.Title.FontSize    = 11;
ax2.XLabel.FontSize   = 11;
%ax2.Title.Position(2) = 1.6;
colormap(ax2,color_mod);
set(ax2,'CLim',ax1.CLim,'Layer','top');
cbar2 = colorbar(ax2,'Location','east');
set(cbar2.Label,'String',{'\textbf{\textsf{Model}}','\textbf{\textsf{$I_{\rm ext}$ (pA)}}'},'Interpreter','latex','FontSize',10,'Rotation',0,'Units','normalized');
cbar2.Position =  [0.2666 0.7579 0.0119 0.1327];
drawnow
cbar2.Label.Position = [1.7460 1.7779 0];
lh=legend(ax2,'Box','off','Location','northeast','FontSize',12);

if saveOutputFigures
    figname = ['D:/Dropbox/p/uottawa/data/mossy_cell_experiment/model_data_paper/theta_vs_ahpmin_',out_fname_suffix];
    fprintf('saving ... %s\n',figname);
    saveFigure(fh, figname, 'png', true, [], 600);
end