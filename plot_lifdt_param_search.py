# -*- coding: utf-8 -*-
import os
import gc
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
#import matplotlib.pyplot as plt
#import matplotlib
import scipy.io
import scipy.optimize
import warnings
import copy
import sys

def main():
    warnings.filterwarnings('ignore')

    # for debug
    #sys.argv.extend(['-paramset','EIFDTBoundSigKLR','-avg','hilo','-avgmodel','same','-I0','0.15:0.4:3'])
    #sys.argv.extend(['-paramset','EIFDTBoundSigKLR','-avg','hilo','-avgmodel','same','-I0','0.15','0.2'])
    #sys.argv.extend(['-paramset','EIFDTBoundSigKLR','-deltaV','12.0'])
    #sys.argv.extend(['-paramset','EIFDT','-thresholdmethod','maxcurvature'])
    #sys.argv.extend(['-paramset','EIFDTBoundSigKLR','-weighted','-I0','0.07:0.4:24'])

    parser = argparse.ArgumentParser(description='Plots LIF and LIFDT neurons')

    # output parameters
    #parser.add_argument('-showinvivo', required=False, action='store_true', default=False, help='shows in vivo data (high current experiments)')
    parser.add_argument('-showall', required=False, action='store_true', default=False, help='shows all figures (otherwise, show only figures related to experimental fit)')
    parser.add_argument('-savedata', required=False, action='store_true', default=False, help='saves data in npz file')
    parser.add_argument('-savedataold', required=False, action='store_true', default=False, help='saves data in npz file for backwards compatibility with old plot py files')
    parser.add_argument('-savemat', required=False, action='store_true', default=False, help='saves data in MAT file')
    parser.add_argument('-avg', nargs=1, required=False, metavar='AVERAGE_PROCEDURE', type=str, default=['all'], choices=['none','all','hilo'], help='[none,all,hilo]; if all, averages over all experimental data currents; if hilo, averages high and low currents separately')
    parser.add_argument('-avgmodel', nargs=1, required=False, metavar='AVERAGE_PROCEDURE', type=str, default=['none'], choices=['same','none','all','hilo'], help='[same,none,all,hilo]; average procedure for model simulation data; if same is used, then it does the same avg procedure as for the experimental data')
    parser.add_argument('-avgrange', nargs=1, required=False, metavar='MIN:MAX', type=inp.input_avgrange_type, default=[''], help='averages the data for currents in [min,max] specified by MIN:MAX to this parameter')
    parser.add_argument('-weighted', required=False, action='store_true', default=False, help='if set, then tries to do a weighted average of the simulation data based on the input currents from experiments (the more input currents in the experimental data, the bigger the weight)')
    parser.add_argument('-showolddata', required=False, action='store_true', default=False, help='shows old data (may be incorrect due to spurious averaging)')
    parser.add_argument('-rescaleI', required=False, action='store_true', default=False, help='rescale input currents of model and experiments to [0,1]')
    
    parser = inp.add_output_parameters(parser,out=['sim'])
    parser = inp.add_neuron_parameters(parser)

    # simulation parameters
    parser = inp.add_simulation_parameters(parser)

    #parser = inp.add_stimulus_parameters(parser,I0=[30,40],tStim=[500.0],DeltaTStim=[2000.0])
    parser = inp.add_stimulus_parameters(parser,I0=['nan'],tStim=[500.0],DeltaTStim=[5000.0])

    args = parser.parse_args()

    # getting figure param
    showOldData = args.showolddata
    showAll = args.showall
    #showInVivo = args.showinvivo
    doWeightCurrents = args.weighted
    saveData = args.savedata
    saveDataOld = args.savedataold
    saveMAT = args.savemat
    saveFig = args.save
    rescaleCurr = args.rescaleI
    outFileNamePrefix = args.out[0]
    outFileFormat = args.format[0]
    outFigResolution = args.dpi[0]
    avgExpData = args.avg[0]
    avgModelData = args.avgmodel[0]
    avgRange = args.avgrange[0]
    thresholdmethod = args.thresholdmethod[0]
    if avgModelData.lower() == 'same':
        avgModelData = avgExpData

    #print(avgRange)
    #print(type(avgRange))

    paramset = args.paramset[0]
    #paramset = 'EIFDTBoundSigKLR'
    #paramset = 'LIFiEIFsfmod1'
    #paramset = 'LIFDTmod3'
    #paramset = 'LIFDTBoundK'
    #paramset = 'LIFDTBoundKLR'
    #avgExpData = 'none'
    #paramset = 'LIFDLTBoundKLRIA'

    neuronArgs = neu.GetDefaultParamSet(paramset,neuronArgs=inp.get_input_neuron_args(args),modified_paramsetArgs=inp.get_cmd_line_neuron_args(args))#inp.get_modified_paramset_args(paramset,args))
    stimArgs = inp.get_input_stimulus_args(args,neuronArgs)
    #stimArgs['I0'] = [0.2]
    #print(neuronArgs)
    #print(stimArgs)

    current_weights = None
    if doWeightCurrents:
        current_weights = inp.get_current_weights_from_volt_traces(stimArgs['I0'],'data/experiments/hMC_StepCurrent_DataJune2021.mat',convert_to_pA=True)

    simArgs = {
                **neuronArgs,
                'ntrials': args.ntrials[0],
                'dt': args.dt[0],
                'noiseSignal': lambda:0.0
              }
    T = args.T[0]

    print('### running neuron with receptive field ...')
    t,I,V,th,g1,g2 = neu.RunSingleNeuron(T,stimArgs,**simArgs)

    print('### running threshold amplitude trials ...')
    if (type(simArgs['I0']) is str) or (not numpy.isscalar(simArgs['I0'])):
        simArgs2 = copy.deepcopy(simArgs)
        if type(simArgs2['I0']) is str:
            simArgs2['I0'] = get_range_from_string(simArgs2['I0'])
        simArgs2['I0'] = numpy.nanmax(simArgs2['I0'])
        print(' ... more than 1 value for input current, using only the greatest value')
    else:
        simArgs2 = simArgs
    DeltaT,th_amp,th_std,theta0 = neu.ProbeThresholdDecay(stimArgs['thDecayDeltaTStim'],50.0,20,thresholdmethod=thresholdmethod,**simArgs2)

    tauTheta = fit_tau_theta(DeltaT,th_amp,th_std,simArgs)
    ISIstruct,AHPMinStruct,SpkThreshStruct,AHPAmpStruct,DeltaThetaStruct,SpkThreshDiffStruct = calc_spike_features(t,I,V,th,stimArgs['I0'],simArgs,tauTheta=tauTheta,Vth=59.0,thresholdmethod=thresholdmethod)
    hasSpk = SpkThreshStruct['avg'].size > 0
    if not hasSpk:
        print('*** WARNING: no spikes detected')

    color_list = p.get_default_colors()

    outFileNamePrefix_fig,outFileNamePrefix_data = output.create_fig_data_dir(outFileNamePrefix,'search',neuronArgs['neuronType'],paramset,saveFig,saveData or saveMAT or saveDataOld)
    if showOldData:
        outFileNamePrefix_fig,outFileNamePrefix_data = (outFileNamePrefix_fig + '_OLD',outFileNamePrefix_data + '_OLD')

    if not showOldData:
        exp_data = inp.import_mc_experiment_matfile('data/experiments/hMC_IntrinsicParameters_normalSteps.mat')

    avg_data_struct = lambda s: inp.avg_data_matrix(s['avg'],s['std'],parVal=s['I'],avgType=avgModelData,axis=1,parValRange=avgRange,weights=current_weights)

    """
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ###
    ###
    ### Plotting Voltage trace
    ###
    ###
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    """

    print('# plotting voltage trace')
    plotTitle = '$V_R=%.2f$mV, $\\tau=%.2f$ms, $\\sigma=%.2f$, $\\theta_0=%.2f$mV, $\\tau_{\\theta}=%.2f$ms, $\\Delta\\theta=%.2f$mV' \
                % (simArgs['Vr'],simArgs['tau'],simArgs['noiseStd'],simArgs['theta0'],simArgs['tauTheta'],simArgs['DeltaTheta'])
    p.plot_complete_voltage_trace(trial_idx=0,simArgs=simArgs,plotTitle=plotTitle,t=t,I=I,V=V,th=th,g1=g1,g2=g2)
    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix_fig,resolution=outFigResolution,file_format=outFileFormat)
    if saveDataOld:
        output.save_data({'t':t,'I':I,'V':V,'th':th,'g1':g1,'g2':g2,'neuronArgs':neuronArgs},outFileNamePrefix_data,file_suffix='_voltage_trace',file_format='npz',numpy=numpy)
    if simArgs['ntrials'] == 1:
        p.plt.show()
        return

    """
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ###
    ###
    ### Plotting voltage distribution
    ###
    ###
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    """
    if showAll and showOldData:
        print('# plotting background voltage distribution')
        VV = remove_spikes(V,59.0) # spikes have V=60.0mV by construction
        _,_,ax,_ = p.plot_noise_distribution(VV,nbins=50)
        p.plt.xlabel('V (mV)')
        p.plt.ylabel('P(V)')
        tit = ax.get_title()
        p.plt.title('Background noise: ' + tit)
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_bgnoise_dist',resolution=outFigResolution,file_format=outFileFormat)
            
    """
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ###
    ###
    ### Plotting threshold increase
    ###
    ###
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    """

    if hasSpk and (showAll and showOldData):
        print('# plotting delta threshold steps')
        if type(I) is list:
            tsStart,tsEnd = find_first_last_idx(I[0]>0)
        else:
            tsStart,tsEnd = find_first_last_idx(I>0)
        thSteps = calc_variation_at_spk(th, V, Vth=59.0, tsStart=tsStart, tsEnd=tsEnd)
        thStepAvg = numpy.nanmean(asarray_nanfill(thSteps),axis=0)
        thStepStd = numpy.nanstd(asarray_nanfill(thSteps),axis=0)
        p.errorfill(numpy.arange(len(thStepAvg))+1, thStepAvg, thStepStd, fmt='o--')
        p.plt.matplotlib.rc('text',usetex=True)
        p.plt.xlabel('Spike #')
        p.plt.ylabel('$\\overline{\\delta\\theta}$ (mV) after spike #')
        p.plt.title('$\\delta\\theta=\\theta(spk_i})-\\theta(spk_{i-1})$')
        p.plt.matplotlib.rc('text',usetex=False)
        p.plt.gca().set_xlim(0,30)
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_delta_threshold_spk_var',resolution=outFigResolution,file_format=outFileFormat)
    
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

    if hasSpk:
        print('# plotting ISI series')
        if showOldData:
            isi_normal_data = pandas.read_table('data/experiments/ISI_normalSteps.dat',sep=' ', header=None, comment='#').to_numpy()
            p.errorfill(isi_normal_data[:,0],isi_normal_data[:,1],isi_normal_data[:,2],fmt=':+', label='100pA-200pA experiments',color=color_list[3])
            p.errorfill(ISIstruct['spk'], ISIstruct['avg'], ISIstruct['std'], fmt='o--', color=color_list[0],ax=p.plt.gca(),label='Model')
            p.plt.xlabel('Spike #')
            p.plt.matplotlib.rc('text',usetex=True)
            p.plt.ylabel('$\\overline{ISI}$ (ms) after spike #')
            p.plt.matplotlib.rc('text',usetex=False)
            p.plt.legend()
            p.plt.gca().set_xlim(0,15)
        else:
            x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_ISI')
            y_exp,yErr_exp,curr_exp = inp.avg_data_matrix(y_exp,yErr_exp,parVal=curr_exp,avgType=avgExpData,axis=1,parValRange=avgRange)
            y_mod,yErr_mod,curr_mod = avg_data_struct(ISIstruct)
            p.plot_param_search_var(x_exp,y_exp,yErr_exp,curr_exp,dict(spk=ISIstruct['spk'],avg=y_mod,std=yErr_mod,I=curr_mod),rescale_currents=rescaleCurr,ylabel_txt='$\\overline{ISI}$ (ms) after spike #',xLim_spk=[0,15])
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_ISI',resolution=outFigResolution,file_format=outFileFormat)
        if saveDataOld:
            output.save_data({'spk_number':ISIstruct['spk'],'ISIavg':ISIstruct['avg'],'ISIstd':ISIstruct['std'],'I':ISIstruct['I'],'neuronArgs':neuronArgs},outFileNamePrefix_data,file_suffix='_ISI',file_format='npz',numpy=numpy)

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

    if hasSpk:
        print('# plotting AHP amplitude')
        if showOldData:
            ahp_normal_data = pandas.read_table('data/experiments/AHP_Amplitude_normalSteps.dat',sep=' ', header=None, comment='#').to_numpy()
            p.errorfill(ahp_normal_data[:,0],ahp_normal_data[:,1],ahp_normal_data[:,2],fmt=':+', label='100pA-200pA experiments',color=color_list[3])
            p.errorfill(AHPAmpStruct['spk'], AHPAmpStruct['avg'], AHPAmpStruct['std'], fmt='o--', color=color_list[0],ax=p.plt.gca(),label='Model')
            p.plt.xlabel('Spike #')
            p.plt.matplotlib.rc('text',usetex=True)
            p.plt.ylabel('AHP amplitude (mV) = $\\theta(spk\\#) - $min(AHP)')
            p.plt.matplotlib.rc('text',usetex=False)
            p.plt.legend()
            p.plt.gca().set_xlim(0,15)
        else:
            x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_AHP_Ampl')
            y_exp,yErr_exp,curr_exp = inp.avg_data_matrix(y_exp,yErr_exp,parVal=curr_exp,avgType=avgExpData,axis=1,parValRange=avgRange)
            y_mod,yErr_mod,curr_mod = avg_data_struct(AHPAmpStruct)
            p.plot_param_search_var(x_exp,y_exp,yErr_exp,curr_exp,dict(spk=AHPAmpStruct['spk'],avg=y_mod,std=yErr_mod,I=curr_mod),rescale_currents=rescaleCurr,ylabel_txt='AHP amplitude (mV) = $\\theta(spk\\#) - $min(AHP)',xLim_spk=[0,15])
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_ahp_amplitude',resolution=outFigResolution,file_format=outFileFormat)
        if saveDataOld:
            output.save_data({'spk_number':AHPAmpStruct['spk'],'ahp_avg':AHPAmpStruct['avg'],'ahp_std':AHPAmpStruct['std'],'I':AHPAmpStruct['I'],'neuronArgs':neuronArgs},outFileNamePrefix_data,file_suffix='_ahp_amplitude',file_format='npz',numpy=numpy)

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

    if hasSpk:
        print('# plotting AHP minimum')
        if showOldData:
            ahp_normal_data = pandas.read_table('data/experiments/AHP_Min_normalSteps.dat',sep=' ', header=None, comment='#').to_numpy()
            p.errorfill(ahp_normal_data[:,0],ahp_normal_data[:,1],ahp_normal_data[:,2],fmt=':+', label='100pA-200pA experiments',color=color_list[3])
            p.errorfill(AHPMinStruct['spk'], AHPMinStruct['avg'], AHPMinStruct['std'], fmt='o--', color=color_list[0],ax=p.plt.gca(),label='Model')
            p.plt.xlabel('Spike #')
            p.plt.matplotlib.rc('text',usetex=True)
            p.plt.ylabel('AHP minimum (mV) after spike #')
            p.plt.matplotlib.rc('text',usetex=False)
            p.plt.legend()
            p.plt.gca().set_xlim(0,15)
        else:
            x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_AHP_Min')
            y_exp,yErr_exp,curr_exp = inp.avg_data_matrix(y_exp,yErr_exp,parVal=curr_exp,avgType=avgExpData,axis=1,parValRange=avgRange)
            y_mod,yErr_mod,curr_mod = avg_data_struct(AHPMinStruct)
            p.plot_param_search_var(x_exp,y_exp,yErr_exp,curr_exp,dict(spk=AHPMinStruct['spk'],avg=y_mod,std=yErr_mod,I=curr_mod),rescale_currents=rescaleCurr,ylabel_txt='AHP minimum (mV) after spike #',xLim_spk=[0,15])
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_ahp_minimum',resolution=outFigResolution,file_format=outFileFormat)
        if saveDataOld:
            output.save_data({'spk_number':AHPMinStruct['spk'],'ahp_avg':AHPMinStruct['avg'],'ahp_std':AHPMinStruct['std'],'I':AHPMinStruct['I'],'neuronArgs':neuronArgs},outFileNamePrefix_data,file_suffix='_ahp_minimum',file_format='npz',numpy=numpy)

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

    if hasSpk:
        print('# plotting DeltaTheta vs. spike #')
        if showOldData:
            dth_normal_data = pandas.read_table('data/experiments/DeltaTheta_normalSteps.dat',sep=' ', header=None, comment='#').to_numpy()
            p.errorfill(dth_normal_data[:,0],dth_normal_data[:,1],dth_normal_data[:,2],fmt=':+', label='100pA-200pA experiments',color=color_list[3])
            p.errorfill(DeltaThetaStruct['spk'], DeltaThetaStruct['avg'], DeltaThetaStruct['std'], fmt='o--', color=color_list[0],ax=p.plt.gca(),label='Model')
            p.plt.xlabel('Spike #')
            p.plt.matplotlib.rc('text',usetex=True)
            p.plt.ylabel('$\\Delta\\theta$ (mV)')
            p.plt.matplotlib.rc('text',usetex=False)
            p.plt.legend()
            p.plt.gca().set_xlim(0,15)
        else:
            x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_DeltaTheta')
            y_exp,yErr_exp,curr_exp = inp.avg_data_matrix(y_exp,yErr_exp,parVal=curr_exp,avgType=avgExpData,axis=1,parValRange=avgRange)
            y_mod,yErr_mod,curr_mod = avg_data_struct(DeltaThetaStruct)
            p.plot_param_search_var(x_exp,y_exp,yErr_exp,curr_exp,dict(spk=DeltaThetaStruct['spk'],avg=y_mod,std=yErr_mod,I=curr_mod),rescale_currents=rescaleCurr,ylabel_txt='$\\Delta\\theta$ (mV)',xLim_spk=[0,15])
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_DeltaTheta_spk_time',resolution=outFigResolution,file_format=outFileFormat)
        if saveDataOld:
            output.save_data({'spk_number':DeltaThetaStruct['spk'],'dthAvg':DeltaThetaStruct['avg'],'dthStd':DeltaThetaStruct['std'],'I':DeltaThetaStruct['I'],'neuronArgs':neuronArgs},outFileNamePrefix_data,file_suffix='_DeltaTheta_spk_time',file_format='npz',numpy=numpy)

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

    if hasSpk:
        print('# plotting threshold at spike timings')
        if showOldData:
            th_normal_data = pandas.read_table('data/experiments/ThresholdDiff_normalSteps.dat',sep=' ', header=None, comment='#').to_numpy()
            p.errorfill(th_normal_data[:,0],th_normal_data[:,1],th_normal_data[:,2],fmt=':+', label='100pA-200pA experiments',color=color_list[3])
            p.errorfill(SpkThreshStruct['spk'], SpkThreshStruct['avg'] - SpkThreshStruct['avg'][0], SpkThreshStruct['std'], fmt='o--', color=color_list[0],ax=p.plt.gca(),label='Model')
            p.plt.matplotlib.rc('text',usetex=True)
            p.plt.xlabel('Spike #')
            p.plt.ylabel('$\\overline{\\theta} - \\theta_0$ (mV) after spike #')
            p.plt.legend()
            p.plt.matplotlib.rc('text',usetex=False)
            p.plt.gca().set_xlim(0,15)
        else:
            x_exp,y_exp,yErr_exp,curr_exp = inp.get_mc_experiment_var(exp_data,'hMC_SpikeThreshold_Diff')
            y_exp,yErr_exp,curr_exp = inp.avg_data_matrix(y_exp,yErr_exp,parVal=curr_exp,avgType=avgExpData,axis=1,parValRange=avgRange)
            y_mod,yErr_mod,curr_mod = avg_data_struct(SpkThreshDiffStruct)
            p.plot_param_search_var(x_exp,y_exp,yErr_exp,curr_exp,dict(spk=SpkThreshDiffStruct['spk'],avg=y_mod,std=yErr_mod,I=curr_mod),rescale_currents=rescaleCurr,ylabel_txt='$\\overline{\\theta} - \\theta_0$ (mV) after spike #',xLim_spk=[0,15])
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_th_spk',resolution=outFigResolution,file_format=outFileFormat)
        if saveDataOld:
            output.save_data({'spk_number':SpkThreshStruct['spk'],'thdiff_spkAvg':SpkThreshDiffStruct['avg'],'th_spkAvg':SpkThreshStruct['avg'],'th_spkStd':SpkThreshStruct['std'],'I':SpkThreshStruct['I'],'neuronArgs':neuronArgs},outFileNamePrefix_data,file_suffix='_th_spk',file_format='npz',numpy=numpy)

    """
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ###
    ###
    ### Plotting threshold decay
    ###
    ###
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    """
    #th_std /= numpy.sqrt(simArgs['ntrials'])
    if hasSpk:
        fitFunc = lambda xx,a,b,c: a * numpy.exp(-b * xx) + c
        fitFuncExp = lambda xx,a,b: a * numpy.exp(-b * xx)
        x_fit = numpy.linspace(DeltaT[0],DeltaT[-1],100)
        th_normal_data = pandas.read_table('data/experiments/MC_RAMP_Tau.dat',
                                    sep=' ', header=None, comment='#').to_numpy()
        fitParam_model, _ = scipy.optimize.curve_fit(fitFunc, DeltaT, th_amp, p0=(1.0,1.0/simArgs['tauTheta'],simArgs['theta0']), maxfev=100000)
        fitParam_data, _ = scipy.optimize.curve_fit(fitFuncExp, th_normal_data[:,0], th_normal_data[:,1], p0=(1.0,0.0001), maxfev=100000)

        print('# plotting threshold decay')
        #p.errorfill(DeltaT, th_amp, th_std, fmt='o--')
        p.errorfill(th_normal_data[:,0], th_normal_data[:,1], th_normal_data[:,2], fmt='o',color=color_list[1], label='Exp. data')
        p.plt.plot(x_fit, fitFuncExp(x_fit,*fitParam_data),'-',linewidth=2,color=color_list[2], label='Exp. data fit: $\\tau_{\\theta}=%.2f$, $\\theta_0=??$'%(1.0/fitParam_data[1]))
        p.errorfill(DeltaT, th_amp - theta0, th_std,color=color_list[0], ax=p.plt.gca(),fmt='o',label='Model data: $\\tau_{\\theta}=%.2f$, $\\theta_0=%.2f$'%(tauTheta,simArgs['theta0']))
        p.plt.plot(x_fit, fitFunc(x_fit,*fitParam_model) - theta0,'-',linewidth=2,color=color_list[3], label='Model fit: $\\tau_{\\theta}=%.2f$, $\\theta_0=%.2f$'%(1.0/fitParam_model[1],fitParam_model[2]))
        p.plt.matplotlib.rc('text',usetex=True)
        p.plt.xlabel('$\\Delta{T}$ (ms) after injected current')
        p.plt.ylabel('$\\theta - \\theta_0$ (mV)')
        p.plt.legend()
        p.plt.title('Fit function: $f(\\Delta T) = a \\exp\\left(-\\Delta T/\\tau_\\theta\\right) + \\theta_0$')
        p.plt.matplotlib.rc('text',usetex=False)
        #p.plt.gca().set_yscale('log')
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_th_decay',resolution=outFigResolution,file_format=outFileFormat)
        if saveDataOld:
            output.save_data({'theta0':theta0,'DeltaT':DeltaT,'th_amp':th_amp,'th_std':th_std,'fitParam':fitParam_model,'neuronArgs':neuronArgs},outFileNamePrefix_data,file_suffix='_th_decay',file_format='npz',numpy=numpy)

    if hasSpk:
        if saveData:
            output.save_data({'ISI':ISIstruct,'AHPMin':AHPMinStruct,'SpkTh':SpkThreshStruct,'AHPAmp':AHPAmpStruct,'DeltaTh':DeltaThetaStruct,'SpkThDiff':SpkThreshDiffStruct,'theta0':theta0,'tauTheta':tauTheta,'stimArgs':stimArgs,'neuronArgs':remove_key(simArgs,'noiseSignal'),'t':t,'I':I,'V':V,'th':th,'g1':g1,'g2':g2},outFileNamePrefix_data,file_suffix='_spk_features',file_format='npz',numpy=numpy)
        if saveMAT:
            data = dict(spk_ISI=ISIstruct['spk'],ISI_avg=ISIstruct['avg'],ISIstd=ISIstruct['std'],
                    spk_AHPAmp=AHPAmpStruct['spk'],AHPAmp_avg=AHPAmpStruct['avg'],AHPAmp_std=AHPAmpStruct['std'],
                    spk_AHPMin=AHPMinStruct['spk'],AHPMin_avg=AHPMinStruct['avg'],AHPMin_std=AHPMinStruct['std'],
                    spk_DeltaTh=DeltaThetaStruct['spk'],DeltaTh_avg=DeltaThetaStruct['avg'],DeltaTh_std=DeltaThetaStruct['std'],
                    spk_SpkTh=SpkThreshStruct['spk'],SpkTh_avg=SpkThreshStruct['avg'],SpkTh_std=SpkThreshStruct['std'],
                    spk_SpkThDiff=SpkThreshDiffStruct['spk'],SpkThDiff_avg=SpkThreshDiffStruct['avg'],SpkThDiff_std=SpkThreshDiffStruct['std'],
                    I_avg=SpkThreshStruct['I'],neuronArgs=neuronArgs,thdecay_DeltaT=DeltaT,thdecay_th=th_amp,thdecay_std=th_std,fitParam=fitParam_model,
                    t=t,I=I,V=V,th=th,g1=g1,g2=g2)
            output.save_data(data,outFileNamePrefix_data,file_suffix='_spk_features',file_format='mat',numpy=numpy)

    p.plt.show()

if __name__ == '__main__':
    main()