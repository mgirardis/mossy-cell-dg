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
import sys

#sys.argv.extend(['.\data_currinj\currinj_spk_feat_EIFDTBoundSigKLR_spk_features.npz','-cmap','plasma','-curr','0'])

parser = argparse.ArgumentParser(description='Plots LIF and LIFDT neurons')
parser = inp.add_output_parameters(parser,out=['currinj_vs_spk'])
parser.add_argument('-cmap', nargs=1, required=False, metavar='COLORMAP', type=str, default=['plasma'], help='colormap for plotting different currents')
parser.add_argument('-curr', nargs='+', required=False, metavar='CURR_SLICE', type=int, default=[-1], help='current number for displaying features vs. spike, if left blank, then defaults to all the currents')
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

curridx = slice(0,None,None) if args.curr[0] == -1 else args.curr
if type(curridx) is list:
    if len(curridx) > 1:
        if len(curridx) == 2:
            curridx = slice(curridx[0],curridx[1],None)
        else:
            curridx = slice(curridx[0],curridx[1],curridx[2])
    else:
        curridx = curridx[0]

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
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'ISI')
curr = [ c*m_curr for c in curr ]
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_ISI')
if type(curridx) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(x_exp,y_exp[:,curridx],curr_exp[curridx],yErr=yErr_exp[:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\overline{ISI}$ (ms)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(x[i],y[i][:,curridx],curr[i][curridx],yErr=yErr[i][:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\overline{ISI}$ (ms)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
        ax[i+1].set_xlim(*ax[0].get_xlim())
    ax[-1].set_xlabel('Spike #')
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ISI',resolution=outFigResolution,file_format=outFileFormat)
else:
    ax = p.plot_two_panel_comparison(x_exp,y_exp[:,curridx],yErr_exp[:,curridx],x,[yy[:,curridx] for yy in y],[yy[:,curridx] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=['{:s}, $I={:g}$'.format(ml,cr[curridx]) for ml,cr in zip(model_label,curr) ],\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['$I={:g}$ {:s}'.format(curr_exp[curridx],unit_curr),None],xlabel_txt=[None,'Spike #'],\
                              ylabel_txt=['$\\overline{ISI}$ (ms) after spike #','$\\overline{ISI}$ (ms) after spike #'])
    ax[1].set_xlim(*ax[0].get_xlim())
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ISI_curr{:g}'.format(curridx),resolution=outFigResolution,file_format=outFileFormat)

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
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'AHPAmp')
curr = [ c*m_curr for c in curr ]
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_AHP_Ampl')
if type(curridx) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(x_exp,y_exp[:,curridx],curr_exp[curridx],yErr=yErr_exp[:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='AHP amp. (mV)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(x[i],y[i][:,curridx],curr[i][curridx],yErr=yErr[i][:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='AHP amp. (mV)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
        ax[i+1].set_xlim(*ax[0].get_xlim())
    ax[-1].set_xlabel('Spike #')
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ahp_amp',resolution=outFigResolution,file_format=outFileFormat)
else:
    ax = p.plot_two_panel_comparison(x_exp,y_exp[:,curridx],yErr_exp[:,curridx],x,[yy[:,curridx] for yy in y],[yy[:,curridx] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=['{:s}, $I={:g}$'.format(ml,cr[curridx]) for ml,cr in zip(model_label,curr) ],\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['$I={:g}$ {:s}'.format(curr_exp[curridx],unit_curr),None],xlabel_txt=[None,'Spike #'],\
                              ylabel_txt=['AHP amplitude (mV) = $\\theta(spk\\#) - $min(AHP)','AHP amplitude (mV) = $\\theta(spk\\#) - $min(AHP)'])
    ax[1].set_xlim(*ax[0].get_xlim())
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ahp_amp_curr{:g}'.format(curridx),resolution=outFigResolution,file_format=outFileFormat)

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
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'AHPMin')
curr = [ c*m_curr for c in curr ]
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_AHP_Min')
if type(curridx) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(x_exp,y_exp[:,curridx],curr_exp[curridx],yErr=yErr_exp[:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='AHP min. (mV)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(x[i],y[i][:,curridx],curr[i][curridx],yErr=yErr[i][:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='AHP min. (mV)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
        ax[i+1].set_xlim(*ax[0].get_xlim())
    ax[-1].set_xlabel('Spike #')
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ahp_min',resolution=outFigResolution,file_format=outFileFormat)
else:
    ax = p.plot_two_panel_comparison(x_exp,y_exp[:,curridx],yErr_exp[:,curridx],x,[yy[:,curridx] for yy in y],[yy[:,curridx] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=['{:s}, $I={:g}$'.format(ml,cr[curridx]) for ml,cr in zip(model_label,curr) ],\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['$I={:g}$ {:s}'.format(curr_exp[curridx],unit_curr),None],xlabel_txt=[None,'Spike #'],\
                              ylabel_txt=['AHP minimum (mV) after spike #','AHP minimum (mV) after spike #'])
    ax[1].set_xlim(*ax[0].get_xlim())
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_ahp_min_curr{:g}'.format(curridx),resolution=outFigResolution,file_format=outFileFormat)

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
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'DeltaTh')
curr = [ c*m_curr for c in curr ]
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_DeltaTheta')
if type(curridx) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(x_exp,y_exp[:,curridx],curr_exp[curridx],yErr=yErr_exp[:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\Delta\\theta$ (mV)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(x[i],y[i][:,curridx],curr[i][curridx],yErr=yErr[i][:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\Delta\\theta$ (mV)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
        ax[i+1].set_xlim(*ax[0].get_xlim())
    ax[-1].set_xlabel('Spike #')
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_DeltaTheta',resolution=outFigResolution,file_format=outFileFormat)
else:
    ax = p.plot_two_panel_comparison(x_exp,y_exp[:,curridx],yErr_exp[:,curridx],x,[yy[:,curridx] for yy in y],[yy[:,curridx] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=['{:s}, $I={:g}$'.format(ml,cr[curridx]) for ml,cr in zip(model_label,curr) ],\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['$I={:g}$ {:s}'.format(curr_exp[curridx],unit_curr),None],xlabel_txt=[None,'Spike #'],\
                              ylabel_txt=['$\\Delta\\theta$ (mV)','$\\Delta\\theta$ (mV)'])
    ax[1].set_xlim(*ax[0].get_xlim())
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_DeltaTheta_curr{:g}'.format(curridx),resolution=outFigResolution,file_format=outFileFormat)

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

n = len(model_data['SpkTh'])
x,y,yErr,curr = inp.get_model_currinj_var(model_data,'SpkTh')
curr = [ c*m_curr for c in curr ]
x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_SpikeThreshold_Diff')
if type(curridx) is slice:
    fig,ax = p.plt.subplots(nrows=n+1,ncols=1,sharex=False,sharey=False,figsize=[6,8])
    p.plt.tight_layout()
    p.plot_curves_with_colormap(x_exp,y_exp[:,curridx],curr_exp[curridx],yErr=yErr_exp[:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\overline{\\theta} - \\theta_0$ (mV)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt='Experiment', ax=ax[0], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
    for i in range(n):
        neuronArgs = model_data['neuronArgs'][i]
        p.plot_curves_with_colormap(x[i],y[i][:,curridx]-model_data['theta0'][i],curr[i][curridx],yErr=yErr[i][:,curridx],cmap_name=cmap_name, \
                            xlabel_txt='',ylabel_txt='$\\overline{\\theta} - \\theta_0$ (mV)', \
                            clabel_txt='$I_{inj}$ (%s)'%(unit_curr),title_txt=model_label[i], ax=ax[i+1], \
                            fmt=':o',markersize=3.3,color_fill='#bbbbbb',alpha_fill=0.3)
        ax[i+1].set_xlim(*ax[0].get_xlim())
    ax[-1].set_xlabel('Spike #')
    p.plt.tight_layout()
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_th_spk',resolution=outFigResolution,file_format=outFileFormat)
else:
    ax = p.plot_two_panel_comparison(x_exp,y_exp[:,curridx],yErr_exp[:,curridx],x,[yy[:,curridx]-extend_vec(model_data['theta0'][i],yy[:,curridx].size) for i,yy in enumerate(y)],[yy[:,curridx] for yy in yErr], \
                              topPlotArgs=dict(fmt=':s',markersize=3.3,linewidth=0.5), \
                              bottomPlotArgs=dict(fmt='o-'), \
                              figArgs=dict(sharex=False,sharey=False), \
                              top_labels='Experiment',bottom_labels=['{:s}, $I={:g}$'.format(ml,cr[curridx]) for ml,cr in zip(model_label,curr) ],\
                              top_color_list=data_color,bottom_color_list=model_color,\
                              title_txt=['$I={:g}$ {:s}'.format(curr_exp[curridx],unit_curr),None],xlabel_txt=[None,'Spike #'],\
                              ylabel_txt=['$\\theta-\\theta_0$ (mV)','$\\theta-\\theta_0$ (mV)'])
    ax[1].set_xlim(*ax[0].get_xlim())
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix,'_th_spk_curr{:g}'.format(curridx),resolution=outFigResolution,file_format=outFileFormat)

p.plt.show()