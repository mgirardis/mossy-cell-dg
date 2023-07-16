import os
import math
import argparse
import numpy
import pandas
#from scipy.io import savemat
import quantities
import modules.input as inp
import modules.output as output
import modules.neurons as neu
import modules.plot as p
from modules.misc import *
import scipy.optimize

parser = argparse.ArgumentParser(description='Plots LIF and LIFDT neurons')
parser = inp.add_output_parameters(parser,out=['currinj_vs_curr'])
parser.add_argument('-cmap', nargs=1, required=False, metavar='COLORMAP', type=str, default=['plasma'], help='colormap for plotting different currents')
parser.add_argument('-spk', nargs='+', required=False, metavar='SPK_SLICE', type=int, default=[-1], help='spk number for displaying features vs. current, if left blank, then defaults to the 10 first spikes')
parser.add_argument('-modelcurrmult', nargs=1, required=False, metavar='MULTIPLIER', type=float, default=[1.0e3], help='multiplies model current by this factor I_new = m * I')
parser.add_argument('-currunit', nargs=1, required=False, metavar='CURRENT_UNIT', type=str, default=['pA'], help='unit for input currents')
parser.add_argument('-choose', required=False, action='store_true', default=False, help='choose input files')
parser.add_argument('files', nargs='*', metavar='SPKFEAT_NPZ_FILE', type=str, default=[''], help='1 up to 3 spk feat files to be plotted')
parser.add_argument('-rescaleI', required=False, action='store_true', default=False, help='rescale input currents of model and experiments to [0,1]')

args = parser.parse_args()

cmap_name = args.cmap[0]
saveFig = args.save
chooseFile = args.choose
outFileNamePrefix = args.out[0]
outFileFormat = args.format[0]
outFigResolution = args.dpi[0]
inp_files = args.files
m_curr = args.modelcurrmult[0]
unit_curr = args.currunit[0]
rescaleCurr = args.rescaleI

spkn = slice(0,10) if args.spk[0] == -1 else args.spk
if type(spkn) is list:
    if len(spkn) > 1:
        if len(spkn) == 2:
            spkn = slice(spkn[0],spkn[1])
        else:
            spkn = slice(spkn[0],spkn[1],spkn[2])
    else:
        spkn = spkn[0]

#spkn = 0

outFileNamePrefix = os.path.join('fig_currinj',outFileNamePrefix)
if not os.path.isdir('fig_currinj'):
    print('*** creating dir: fig_currinj')
    os.mkdir('fig_currinj')

exp_data = inp.import_mc_experiment_matfile('D:\\Dropbox\\p\\uottawa\\data\\mossy_cell_experiment\\by_current\\hMC_IntrinsicParameters_normalSteps.mat')

if chooseFile:
    model_data = inp.import_model_currinj(inp.get_files_GUI('Select up to 3 spike feature files...','./data_currinj'),allow_pickle=True)
else:
    if (len(inp_files) == 1) and (len(inp_files[0]) == 0):
        print(' ... using some default files, if they exist')
        model_data = inp.import_model_currinj(['data_currinj\\currinj_spk_feat_LIFDTmod3_spk_features.npz',
                                            'data_currinj\\currinj_spk_feat_LIFDTK_spk_features.npz',
                                            'data_currinj\\currinj_spk_feat_LIFDTBoundK_spk_features.npz'],allow_pickle=True)
    else:
        model_data = inp.import_model_currinj(inp_files,allow_pickle=True)

model_label = [ data['neuronType'] for data in model_data['neuronArgs'] ]

color_list = p.get_default_colors()
data_color = color_list[3]
model_color = [c for k,c in enumerate(color_list) if k != 3 and k != 1]
model_color[0],model_color[2] = model_color[2],model_color[0] # swapping blue for violet
model_color[2],model_color[5] = model_color[5],model_color[2] # swapping blue for gray
model_color[2] = '#000000' # instead of gray, use black

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Plotting ISI
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

n = len(model_data['ISI'])
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'ISI',transpose=True)
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_ISI',transpose=True)
if type(spkn) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(curr_exp,y_exp[:,spkn],x_exp[spkn],yErr=yErr_exp[:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\overline{ISI}$ (ms)', \
                            clabel_txt='Spike #',title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(curr[i]*m_curr,y[i][:,spkn],x[i][spkn],yErr=yErr[i][:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\overline{ISI}$ (ms)', \
                            clabel_txt='Spike #',title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    ax[-1].set_xlabel('Injected step current (%s)'%unit_curr)
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ISI',resolution=outFigResolution,file_format=outFileFormat)
else:
    rescale_factor = [0,1] if rescaleCurr else rescaleCurr
    x_label = 'Rescaled injected current (step)' if rescaleCurr else 'Injected step current (%s)'%unit_curr
    x,y,yErr,curr = inp.get_model_currinj_var(model_data,'ISI',transpose=True,rescale_current=rescale_factor)
    curr = [ c*m_curr for c in curr ]
    p.plot_two_panel_comparison(curr_exp,y_exp[:,spkn],yErr_exp[:,spkn],curr,[yy[:,spkn] for yy in y],[yy[:,spkn] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=model_label,\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['spike # %d'%(spkn),None],xlabel_txt=[None,x_label],\
                              ylabel_txt=['$\\overline{ISI}$ (ms) after spike #','$\\overline{ISI}$ (ms) after spike #'])
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ISI_spk%d'%(spkn),resolution=outFigResolution,file_format=outFileFormat)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Plotting AHP Amplitude
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

n = len(model_data['AHPAmp'])
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'AHPAmp',transpose=True)
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_AHP_Ampl',transpose=True)
if type(spkn) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(curr_exp,y_exp[:,spkn],x_exp[spkn],yErr=yErr_exp[:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='AHP amp. (mV)', \
                            clabel_txt='Spike #',title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(curr[i]*m_curr,y[i][:,spkn],x[i][spkn],yErr=yErr[i][:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='AHP amp. (mV)', \
                            clabel_txt='Spike #',title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    ax[-1].set_xlabel('Injected step current (%s)'%unit_curr)
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ahp_amp',resolution=outFigResolution,file_format=outFileFormat)
else:
    rescale_factor = [0,1] if rescaleCurr else rescaleCurr
    x_label = 'Rescaled injected current (step)' if rescaleCurr else 'Injected step current (%s)'%unit_curr
    x,y,yErr,curr = inp.get_model_currinj_var(model_data,'AHPAmp',transpose=True,rescale_current=rescale_factor)
    curr = [ c*m_curr for c in curr ]
    p.plot_two_panel_comparison(curr_exp,y_exp[:,spkn],yErr_exp[:,spkn],curr,[yy[:,spkn] for yy in y],[yy[:,spkn] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=model_label,\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['spike # %d'%(spkn),None],xlabel_txt=[None,x_label],\
                              ylabel_txt=['AHP amplitude (mV)','AHP amplitude (mV)'])
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ahp_amp_spk%d'%(spkn),resolution=outFigResolution,file_format=outFileFormat)
"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Plotting AHP Minimum
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

n = len(model_data['AHPMin'])
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'AHPMin',transpose=True)
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_AHP_Min',transpose=True)
if type(spkn) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(curr_exp,y_exp[:,spkn],x_exp[spkn],yErr=yErr_exp[:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='AHP min. (mV)', \
                            clabel_txt='Spike #',title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(curr[i]*m_curr,y[i][:,spkn],x[i][spkn],yErr=yErr[i][:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='AHP min. (mV)', \
                            clabel_txt='Spike #',title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    ax[-1].set_xlabel('Injected step current (%s)'%unit_curr)
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ahp_min',resolution=outFigResolution,file_format=outFileFormat)
else:
    rescale_factor = [0,1] if rescaleCurr else rescaleCurr
    x_label = 'Rescaled injected current (step)' if rescaleCurr else 'Injected step current (%s)'%unit_curr
    x,y,yErr,curr = inp.get_model_currinj_var(model_data,'AHPMin',transpose=True,rescale_current=rescale_factor)
    curr = [ c*m_curr for c in curr ]
    p.plot_two_panel_comparison(curr_exp,y_exp[:,spkn],yErr_exp[:,spkn],curr,[yy[:,spkn] for yy in y],[yy[:,spkn] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=model_label,\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['spike # %d'%(spkn),None],xlabel_txt=[None,x_label],\
                              ylabel_txt=['AHP minimum (mV) after spike #','AHP minimum (mV) after spike #'])
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ahp_min_spk%d'%(spkn),resolution=outFigResolution,file_format=outFileFormat)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Plotting DeltaThreshold
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

n = len(model_data['DeltaTh'])
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'DeltaTh',transpose=True)
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_DeltaTheta',transpose=True)
if type(spkn) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(curr_exp,y_exp[:,spkn],x_exp[spkn],yErr=yErr_exp[:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\Delta\\theta$ (mV)', \
                            clabel_txt='Spike #',title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(curr[i]*m_curr,y[i][:,spkn],x[i][spkn],yErr=yErr[i][:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\Delta\\theta$ (mV)', \
                            clabel_txt='Spike #',title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    ax[-1].set_xlabel('Injected step current (%s)'%unit_curr)
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_DeltaTheta',resolution=outFigResolution,file_format=outFileFormat)
else:
    rescale_factor = [0,1] if rescaleCurr else rescaleCurr
    x_label = 'Rescaled injected current (step)' if rescaleCurr else 'Injected step current (%s)'%unit_curr
    x,y,yErr,curr = inp.get_model_currinj_var(model_data,'DeltaTh',transpose=True,rescale_current=rescale_factor)
    curr = [ c*m_curr for c in curr ]
    p.plot_two_panel_comparison(curr_exp,y_exp[:,spkn],yErr_exp[:,spkn],curr,[yy[:,spkn] for yy in y],[yy[:,spkn] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=model_label,\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['spike # %d'%(spkn),None],xlabel_txt=[None,x_label],\
                              ylabel_txt=['$\\Delta\\theta$ (mV)','$\\Delta\\theta$ (mV)'])
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_DeltaTheta_spk%d'%(spkn),resolution=outFigResolution,file_format=outFileFormat)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Plotting spike threshold
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

#if type(spkn) is int:
#    spkn += 1
n = len(model_data['SpkThDiff'])
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'SpkThDiff',transpose=True)
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_SpikeThreshold_Diff',transpose=True)
if type(spkn) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(curr_exp,y_exp[:,spkn],x_exp[spkn],yErr=yErr_exp[:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\overline{\\theta} - \\theta_0$ (mV)', \
                            clabel_txt='Spike #',title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        #yy = [ y[i][:,spkn][j]-model_data['theta0'][i][j] for j in range(len(curr[i])) ] # subtracting theta0 from SpkTh for each current in curr[i] for model i
        p.plot_curves_with_colormap(curr[i]*m_curr,y[i][:,spkn],x[i][spkn],yErr=yErr[i][:,spkn],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\overline{\\theta} - \\theta_0$ (mV)', \
                            clabel_txt='Spike #',title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    ax[-1].set_xlabel('Injected step current (%s)'%unit_curr)
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_th_spk',resolution=outFigResolution,file_format=outFileFormat)
else:
    rescale_factor = [0,1] if rescaleCurr else rescaleCurr
    x_label = 'Rescaled injected current (step)' if rescaleCurr else 'Injected step current (%s)'%unit_curr
    x,y,yErr,curr = inp.get_model_currinj_var(model_data,'SpkTh',transpose=True,rescale_current=rescale_factor)
    curr = [ c*m_curr for c in curr ]
    x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_SpikeThreshold',transpose=True)
    p.plot_two_panel_comparison(curr_exp,y_exp[:,spkn],yErr_exp[:,spkn],curr,[yy[:,spkn] for yy in y],[yy[:,spkn] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=model_label,\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['spike # %d'%(spkn),None],xlabel_txt=[None,x_label],\
                              ylabel_txt=['$\\theta_0$ (mV)','$\\theta_0$ (mV)'])
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_th_spk%d'%(spkn),resolution=outFigResolution,file_format=outFileFormat)

p.plt.show()