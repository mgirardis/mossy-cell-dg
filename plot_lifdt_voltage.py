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
import scipy.io
import sys

def main():
    
    # for debug
    #sys.argv.extend(['-paramset','EIFDTBoundSigKLR','-tauRiseTheta','1'])
    #sys.argv.extend(['-paramset','AdEIFDTBoundSigKLR'])
    #sys.argv.extend(['-paramset','EIFDT'])
    #sys.argv.extend(['-paramset','EIFDTBoundKLR','-rv','0','-deltaV','-15','-DeltaGK','0'])
    #sys.argv.extend(['-paramset','EIFDTBoundSigKLRIA'])
    #sys.argv.extend(['-paramset','EIFDTBoundSigKLR'])

    parser = argparse.ArgumentParser(description='Plots LIF and LIFDT neurons')

    # output parameters
    parser.add_argument('-showahpexper', required=False, action='store_true', default=False, help='shows experiment of the AHP min vs. adaptive current parameters')
    parser.add_argument('-showdetail', required=False, action='store_true', default=False, help='shows threshold dynamics in detail')
    parser.add_argument('-savedata', required=False, action='store_true', default=False, help='saves data in MAT file')

    parser = inp.add_output_parameters(parser,out=['sim'])
    parser = inp.add_neuron_parameters(parser)

    # DeltaGK experiment
    parser.add_argument('-par', nargs=1, required=False, metavar='VALUE', type=str,default=['DeltaGK'], help='parameter name to vary for AHP experiment')
    parser.add_argument('-parRange', nargs=1,required=False,metavar='RANGE_STR',type=str,default=['0.1:1.5:15'],help='range string of type INIT:END:AMOUNT')

    # simulation parameters
    parser = inp.add_simulation_parameters(parser,T=[3000.0],dt=[0.2],ntrials=[1])

    # stimulus parameters
    parser = inp.add_stimulus_parameters(parser,I0=['nan'],tStim=[500.0],DeltaTStim=[2000.0])
    #parser = inp.add_stimulus_parameters(parser,I0=[30,40],tStim=[500.0],DeltaTStim=[2000.0])

    args = parser.parse_args()

    # getting param values
    T = args.T[0]

    showAHPExperiment = args.showahpexper
    showDetail = args.showdetail
    saveData = args.savedata
    saveFig = args.save
    outFileNamePrefix = args.out[0]
    outFileFormat = args.format[0]
    outFigResolution = args.dpi[0]

    paramset = args.paramset[0]
    #paramset = 'EIFDTBoundSigKLR'
    #paramset = 'LIFDTVBoundKLR'
    #paramset = 'LIFDTK'
    #paramset = 'LIFDLTBoundKLRIA'

    neuronArgs = neu.GetDefaultParamSet(paramset,neuronArgs=inp.get_input_neuron_args(args),modified_paramsetArgs=inp.get_cmd_line_neuron_args(args))#inp.get_modified_paramset_args(paramset,args))
    neuronType = neuronArgs['neuronType']

    stimArgs = inp.get_input_stimulus_args(args,neuronArgs)

    #print(neuronArgs)
    #print(stimArgs)

    tStim = stimArgs['tStim']
    DeltaTStim = stimArgs['DeltaTStim']

    simArgs = {     **neuronArgs,
                    'ntrials': args.ntrials[0],
                    'dt': args.dt[0],
                    'noiseSignal': lambda:0.0 }

    t,I,V,th,g1,g2 = neu.RunSingleNeuron(T,stimArgs,**simArgs)
    #print('V={:g};    theta={:g};    m={:g}'.format(V[0][-1],th[0][-1],g2[0][-1]))
    
    color_list = p.get_default_colors()

    outFileNamePrefix_fig,outFileNamePrefix_data = output.create_fig_data_dir(outFileNamePrefix,'voltage',neuronArgs['neuronType'],paramset,saveFig,saveData)

    plotTitle = '$V_R=%.2f$mV, $\\tau=%.2f$ms, $\\sigma=%.2f$, $\\theta_0=%.2f$mV, $\\tau_{\\theta}=%.2f$ms, $\\Delta\\theta=%.2f$mV' \
                % (neuronArgs['Vr'],neuronArgs['tau'],neuronArgs['noiseStd'],neuronArgs['theta0'],neuronArgs['tauTheta'],neuronArgs['DeltaTheta'])
    ax = p.plot_complete_voltage_trace(trial_idx=0,simArgs=simArgs,plotTitle=plotTitle,t=t,I=I,V=V,th=th,g1=g1,g2=g2)
    if saveFig:
        fn1 = output.save_fig(p.plt,outFileNamePrefix_fig,resolution=outFigResolution,file_format=outFileFormat)

    if showAHPExperiment:
        outFileNamePrefix_data += '_' + args.par[0]
    dataFileName = output.check_and_get_filename(outFileNamePrefix_data + '.mat')
    if saveData:
        output.save_data({'t':t,'I':I,'V':V,'th':th,**simArgs},dataFileName,file_format='mat')

    if showDetail:
        p.plot_horizontal_lines([simArgs['Vb'],simArgs['Vr']],ax=ax[0],color=(0,0,0,1),linestyle='--',linewidth=0.5)
        p.plot_horizontal_lines(simArgs['theta0'],ax=ax[1],color=(0,0,0,1),linestyle='--',linewidth=0.8)
        ax[1].vlines(tStim+60,simArgs['theta0'],simArgs['theta0']+simArgs['DeltaTheta'],linestyle='--',linewidth=0.8,color=(0,0,0,1))
        p.plt.matplotlib.rc('text',usetex=True)
        ax[0].text(t[0]+30,simArgs['Vb']+5, 'Baseline potential, $V_b$', fontsize=10)
        ax[0].text(t[0]+30,simArgs['Vr']+5, 'Reset potential, $V_r$', fontsize=10)
        ax[1].text(t[0]+30,simArgs['theta0']+0.5, 'Initial threshold, $\\theta_0$', fontsize=10)
        ax[1].text(tStim+40,simArgs['theta0']+1, 'Threshold step, $\\Delta\\theta$', fontsize=10, ha='right')
        p.plt.matplotlib.rc('text',usetex=False)

        if saveFig:
            #fn1 = output.check_and_get_filename(outFileNamePrefix + '_' + paramset + '.' + outFileFormat)#overwrite last fn1 file
            output.save_fig(p.plt,fn1,resolution=outFigResolution,file_format=outFileFormat)

        theta0 = simArgs['theta0']
        tauTheta = simArgs['tauTheta']
        spk_times = numpy.nonzero(V[0]>59.0)[0]
        n_spk = len(spk_times)
        ISI = numpy.diff(t[spk_times])
        t_spk = t[spk_times[:-1]]
        th_bef_spk = calc_theta_before_spk(th[0][spk_times],theta0,tauTheta,ISI)
        fig = p.plt.figure()
        aax = fig.gca()
        aax.plot(t,th[0],linewidth=1,color=color_list[2])
        aax.set_xlim(tStim,tStim+DeltaTStim)
        [ aax.vlines(tt+10,th0,th0+simArgs['DeltaTheta'],linestyle='--',linewidth=0.8,color=(0,0,1,1)) for tt,th0 in zip(t[spk_times-1],th[0][spk_times-1]) ]
        p.plot_horizontal_lines(th[0][spk_times],ax=aax,linestyle='--',linewidth=0.5,color=(0,0,0,1))
        p.plot_horizontal_lines(simArgs['theta0'],ax=aax,linestyle='--',linewidth=0.5,color=(0,0,0,1))
        [ aax.hlines(th0,t0,t0+isi0,linestyle='--',linewidth=0.8,color=(1,0,0,1)) for th0,t0,isi0 in zip(th_bef_spk,t_spk,ISI) ]

        p.plt.matplotlib.rc('text',usetex=True)
        aax.set_ylabel('Threshold $\\theta$, mV')
        [ aax.text(tt+20,th0+simArgs['DeltaTheta']/2.0-0.4,'$\\Delta\\theta$',fontsize=12,ha='left',color=(0,0,1,1)) for tt,th0 in zip(t[spk_times-1],th[0][spk_times-1]) ]
        [ aax.text(t0+isi0/2+10,th0-0.2,'$ISI_%d$'%(i+1),fontsize=12,ha='left',va='top',color=(1,0,0,1)) for th0,t0,isi0,i in zip(th_bef_spk,t_spk,ISI,range(len(ISI))) ]
        theta_label = [ '$\\theta_%d$'%i for i in range(n_spk) ]
        p.label_point(t[spk_times-1],th[0][spk_times-1],theta_label,ax=aax,plotArgs={'color':(0,0,0,1),'linestyle':'none','markersize':3},fontsize=14,ha='right',va='bottom')
        p.plt.matplotlib.rc('text',usetex=False)
        aax.set_xlabel('Time (ms)')
        
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_threshold_detail',resolution=outFigResolution,file_format=outFileFormat)

    #showAHPExperiment = True
    if showAHPExperiment:
        #simArgs = {**neu.lifdtkArgs,
        simArgs = {**neuronArgs,
                'dt':simArgs['dt'],
                'noiseSignal': lambda:0.0}
        parName = args.par[0]
        outFileNamePrefix_fig += '_' + parName
        
        par_values = get_range_from_string(args.parRange[0])#numpy.linspace(args.par1[0],args.par2[0],args.npar[0])
        t,V,g1,g2,th = neu.RunSimOverParamRange(parName,par_values,T=T,**simArgs)
        tspk_idx = numpy.asarray([ tt[0] for tt in get_values_at_spike([numpy.arange(len(t)) for k in range(len(V))],V,Vth=59.0) ])
        tspk = numpy.asarray([ tt[0] for tt in get_values_at_spike(t,V,Vth=59.0) ])
        ahp_min,t_min = calc_min_between_instants(V,tsEnd=tspk_idx)
        ahp_min = numpy.asarray(ahp_min)
        t_min = numpy.asarray(t_min) * simArgs['dt']

        p.plot_two_panel_comparison(t,numpy.asarray(V).T,None,t,numpy.asarray(g1).T,None,figArgs=dict(sharex=True),top_labels=['{:s} = {:g}'.format(parName,v) for v in par_values],xlabel_txt=[None,'Time (ms)'],ylabel_txt=['Voltage','g_1'],topPlotArgs=dict(fmt='-'),bottomPlotArgs=dict(fmt='-'),showLegend=[1,0],xLim=[[0,tspk[0]+tspk[-1]],None],legendArgs=dict(bbox_to_anchor=(1,1),loc='upper right',fontsize='xx-small'))
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_ahpexper_V',resolution=outFigResolution,file_format=outFileFormat)

        p.plot_many_panels([par_values,par_values,par_values,t_min,tspk,tspk],
                         [ahp_min,t_min,tspk,ahp_min,ahp_min,t_min],[None,None,None,None,None,None],plotArgs=dict(fmt='o:'),figArgs=dict(nrows=3,ncols=2),
                         labels=[None,None,None,None,None,None],title_txt=[simArgs['neuronType'],None,None,None,None,None],
                         xlabel_txt=[parName,parName,parName,'$t_{min}$ (ms)','$ISI_1$ (ms)','$ISI_1$ (ms)'],
                         ylabel_txt=['$V_{min}$ (mV)','$t_{min}$ (ms)', '$ISI_1$ (ms)','$V_{min}$ (mV)','$V_{min}$ (mV)','$t_{min}$ (ms)'],showLegend=0)
        p.plt.tight_layout()
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'_ahpexper_AHP_feat',resolution=outFigResolution,file_format=outFileFormat)
            
        if saveData:
            output.save_data({'ahp_min':ahp_min,'t_min':t_min,'V_par':numpy.asarray(V).T,'t_par':t,'gK_par':numpy.asarray(g1).T,'par_values':par_values,'par':parName,'tspk':tspk},dataFileName,file_format='mat',append=True)

    p.plt.show()

if __name__ == '__main__':
    main()