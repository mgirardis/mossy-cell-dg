clearvars
close all

addpath func

% all_cells = load('../2020-06-25/hMC_StepCurrent_Data.mat');
all_cells = load('../2020-06-25/hMC_StepCurrent_DataJune2021.mat');
cells = all_cells;
% cell = 'Cell1_20180418R20'
% cell = 'Cell1_20180418R21'
% cell = 'Cell1_20180418R30'
% cell = 'Cell2_20180425R03'
% cell = 'Cell2_20180425R04'
% cell = 'Cell2_20180425R10'
% cell = 'Cell3_20180516R03'
% cell = 'Cell3_20180516R05'
% cell = 'Cell4_20180516R14'
% cell = 'Cell4_20180516R15'
% cell = 'Cell4_20180516R24'
% cell = 'Cell5_20181023R14'
% cell = 'Cell6_20190328R41'
% cell = 'Cell7_20190827R06'
% cell = 'Cell8_20190906R05'
% cell = 'Cell8_20190906R33'
% cell = 'Cell8_20190906R38'
% cell = 'Cell9_T20181011R03'
% cell = 'Cell9_T20181011R04'
% cell = 'Cell9_T20181011R06'
% cell = 'Cell10_20191126R14'
% cell = 'Cell11_20200109R03'
% cell = 'Cell11_20200109R04'
% cell = 'Cell11_20200109R05'
% cell = 'Cell11_20200109R08'
% cell = 'Cell11_20200109R09'
% cell = 'Cell11_20200109R11'
% cell = 'Cell11_20200109R13'
% cell = 'Cell12_20200305R07'
% cell = 'Cell13_20200305R07'


all_cells_joined = join_same_cells_side_by_side(all_cells);
% cells = all_cells_joined;
% cell = 'Cell1'; % early/late spiking
% cell = 'Cell2'; % late spiking
% cell = 'Cell3'; % early/late spiking
% cell = 'Cell4'; % very early spiking
% cell = 'Cell5'; % late spiking
% cell = 'Cell6'; % late spiking
% cell = 'Cell7'; % late spiking
% cell = 'Cell8'; % very early spiking
% cell = 'Cell9'; % late spiking
% cell = 'Cell10'; % late spiking
% cell = 'Cell11'; % early spiking
% cell = 'Cell12'; % early/late spiking

%%

ax = axes;
hold(ax,'on');

I_inj = cells.(cell)(1,2:end);

[I_inj,k_I] = sort(I_inj);
V = cells.(cell)(2:end,k_I+1); % +1 because the data starts at column 2
time = cells.(cell)(2:end,1);

n_curves = size(V,2);
colors = winter(n_curves);
for i = 1:n_curves
    plot(ax,time,V(:,i),'-','DisplayName',sprintf('I = %g pA',I_inj(i)),'Color',colors(i,:));
end
xlabel(ax,'Time (ms)');
ylabel(ax,'Voltage (mV)');
legend(ax,'Location','eastoutside');
