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
import modules.misc as misc
import scipy.optimize
import sys


def main():
    # for debug
    #sys.argv.extend(['-paramset', 'EIFDTBoundSigKLR', '-spk', '0', '3'])
    #sys.argv.extend(['-paramset', 'EIFDT', '-thresholdmethod', 'minderivative'])
    #sys.argv.extend(['-paramset', 'LIFex'])
    #sys.argv.extend(['-paramset', 'EIFDT', '-DeltaT', '1.0', '-VB', '-67.0', '-VT', '-40.0', '-DeltaTheta','0.0','-thetaInit','-40.0'])
    #sys.argv.extend(['-paramset', 'EIFDT', '-thresholdmethod', 'minderivative', '-Vpeak', '20'])
    #sys.argv.extend(['-paramset', 'EIFSubDT', '-thresholdmethod', 'minderivative', '-noise_stddev', '0.0'])

    parser = argparse.ArgumentParser(description='Plots LIF and LIFDT neurons')
    parser = inp.add_output_parameters(parser,out=['theta_I_curve'])
    parser.add_argument('-savedata', required=False, action='store_true', default=False, help='saves data in npz file')
    parser.add_argument('-saveexpdata', required=False, action='store_true', default=False, help='experimental data theta-I curve calculation')
    parser.add_argument('-rescaleI', required=False, action='store_true', default=False, help='rescale input currents of model and experiments to [0,1]')
    parser.add_argument('-showvoltage', required=False, action='store_true', default=False, help='shows voltage plots for each current')
    parser = inp.add_neuron_parameters(parser)
    parser = inp.add_simulation_parameters(parser,T=[1000.0])
    parser.add_argument('-spk', nargs='+', required=False, metavar='SPK_SLICE', type=int, default=[0], help='spk number for displaying features vs. current, if set to -1, then defaults to the 10 first spikes (other int values plot only that spike); if list o int, then create a slice with elements [s0,s1,nspks]')
    parser = inp.add_stimulus_parameters(parser,I0=['nan'],tStim=[100.0],DeltaTStim=[800.0])
    #parser = inp.add_stimulus_parameters(parser,I0=['0.05:0.35:10'],tStim=[500.0],DeltaTStim=[5000.0])
    #parser.add_argument('-cmap', nargs=1, required=False, metavar='COLORMAP', type=str, default=['jet'], help='colormap for plotting different currents')
    #parser.add_argument('-choose', required=False, action='store_true', default=False, help='choose input files')

    args = parser.parse_args()
    saveData = args.savedata
    saveExpData = args.saveexpdata
    saveFig = args.save
    showVoltage = args.showvoltage
    rescaleCurr = args.rescaleI
    outFileNamePrefix = args.out[0]
    outFileFormat = args.format[0]
    outFigResolution = args.dpi[0]
    thresholdmethod = args.thresholdmethod[0]

    paramset = args.paramset[0]
    #paramset = 'LIFDLTBoundKLR'

    dt = args.dt[0]
    T = args.T[0]
    ntrials = args.ntrials[0]

    spkn = slice(0,10) if args.spk[0] == -1 else args.spk
    if type(spkn) is list:
        if len(spkn) > 1:
            if len(spkn) == 2:
                spkn = slice(spkn[0],spkn[1])
            else:
                spkn = slice(spkn[0],spkn[1],spkn[2])
        else:
            spkn = spkn[0]

    # debug params
    #paramset = 'LIFDLTBoundKLR'
    #ntrials = 2

    outFileNamePrefix_fig,outFileNamePrefix_data = output.create_fig_data_dir(outFileNamePrefix + '_' + paramset,'initthresh','','',saveFig,saveData)

    stimParam = inp.get_input_stimulus_args(args=args,defaultI0='nan')
    if type(stimParam['I0']) is str:
        if stimParam['I0'] == 'nan':
            stimParam['I0'] = neu.GetDefaultCurrentRange(paramset)

    # debug params
    #stimParam['I0'] = numpy.linspace(stimParam['I0'][0],stimParam['I0'][-1],3)

    simParam = dict(**neu.GetDefaultParamSet(paramset,modified_paramsetArgs=inp.get_cmd_line_neuron_args(args)),dt=dt,noiseSignal=lambda:0.0,ntrials=ntrials)


    """
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ###
    ###
    ###  calculating experimental curves
    ###
    ###
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    """
    exp_data = inp.import_mc_experiment_matfile('data/experiments/hMC_IntrinsicParameters_normalSteps.mat')
    print('   ... loading experimental thresholds')
    _,th_exp,therr_exp,I_exp = inp.get_mc_experiment_var(exp_data,'hMC_SpikeThreshold',transpose=True)
    Iall_exp = misc.expand_vec_into_mat(I_exp,th_exp.shape[1],axis=1).flatten() # all values of currents corresponding to ISI are sequentially arranged
    thall_exp = th_exp.flatten() # all ISI values are sequentially arranged
    if rescaleCurr:
        I_exp_rescaled,coeff_exp = misc.linearTransf(I_exp,[0,1],returnCoeff=True)
    else:
        I_exp_rescaled = I_exp
        coeff_exp = (0.0,1.0)
    if saveExpData:
        expFileName = 'data/experiments/hMC_theta_I_curve'
        output.save_data(dict(I=I_exp,Iall=Iall_exp,th_all=thall_exp,spkn=numpy.arange(th_exp.shape[1])[spkn],theta=th_exp[:,spkn],theta_err=therr_exp[:,spkn],th=th_exp,th_err=therr_exp,linearTransfCoeff=coeff_exp,Irescaled=I_exp_rescaled,I_resc_func='Ir=c1+c2*I'),expFileName,file_format='mat',numpy=numpy)


    mc_lat_data = scipy.io.loadmat('data/experiments/hMC_StepCurrent_1stLatency.mat')
    I_mc_for_lat = mc_lat_data['hMC_1stLatency_AVG'][:,0].flatten() # converting from pA to nA
    t0spk_mc = mc_lat_data['hMC_1stLatency_AVG'][:,1].flatten() # t to first spk (latency) in ms
    t0spk_err_mc = mc_lat_data['hMC_1stLatency_AVG'][:,2].flatten()

    """
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ###
    ###
    ###  running simulations
    ###
    ###
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    """
    print('... {:s}: {:d} trials, I0 = {}'.format(simParam['neuronType'],ntrials,stimParam['I0']))
    t,_,V,th,_,_ = neu.RunSingleNeuron(T,stimParam,**simParam)
    del _

    """
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ###
    ###
    ###  calculating model curves
    ###
    ###
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    """
    print('   ... calculating model thresholds')
    if numpy.isscalar(stimParam['I0']):
        raise ValueError('you have to simulate for more than one injected current value')


    # debug
    # thresholdmethod = 'model'
    if (thresholdmethod == 'model') and neu.has_no_threshold(paramset):
        print('WARNING ::: Using thresholdmethod == minderivative')
        thresholdmethod = 'minderivative'
    if thresholdmethod == 'model':
        th_model_per_sim = [misc.asarray_nanfill(misc.get_values_at_spike(th[i],V[i],Vth=59.0,tsDelay=-1)) for i in range(len(V))]
        tspk_model_per_sim = [misc.asarray_nanfill(misc.get_values_at_spike(t,V[i],Vth=59.0)) for i in range(len(V))]
    else:
        findpeaks_args = dict(prominence=50.0)
        dt = t[1] - t[0]
        if thresholdmethod == 'maxcurvature':
            res = [misc.calc_threshold_max_curvature(V[i],squeeze_len1_list_input=False,squeeze_len1=False,nPoints=int(6.0/dt),use_findpeaks=True,V0=50.0,return_time_idx=True,**findpeaks_args) for i in range(len(V))]
        else:
            res = [misc.calc_threshold_min_derivative(V[i],squeeze_len1_list_input=False,squeeze_len1=False,nPoints=int(6.0/dt),use_findpeaks=True,V0=50.0,return_time_idx=True,**findpeaks_args) for i in range(len(V))]
        th_model_per_sim = [misc.asarray_nanfill(rr[0]) for rr in res]
        tspk_model_per_sim = [misc.asarray_nanfill(rr[1])*dt for rr in res]

    th_model = misc.asarray_nanfill([numpy.nanmean(th_model_per_sim[i],axis=0) for i in range(len(th_model_per_sim))])
    thstd_model = misc.asarray_nanfill([numpy.nanstd(th_model_per_sim[i],axis=0) for i in range(len(th_model_per_sim))])
    I_model = numpy.asarray([stimParam['I0']] if numpy.isscalar(stimParam['I0']) else stimParam['I0'])

    tspk_lat_avg = numpy.asarray([ numpy.nanmean(tspk_model_per_sim[k][:,0]-stimParam['tStim']) for k in range(len(tspk_model_per_sim)) ])
    tspk_lat_std = numpy.asarray([ numpy.nanstd(tspk_model_per_sim[k][:,0]-stimParam['tStim']) for k in range(len(tspk_model_per_sim)) ])

    # we want to fill in the experimental data that correspond to f=0 in the model with f=0 in the experiment as well
    # for that we first find the I index in which the f>0 in the model
    # and only use k:end as indices of I_model to calculate the transform
    #k = int(len(I_model) / 2)
    k = 0

    # I want both the I_exp_rescaled and the I_model_rescaled to vary according to the same rate
    # this is to avoid bias on the comparison of models towards the experimental data
    # then I impose dIexp,r = dImod,r
    # knowing that dIexp,r/dIexp = coeff_exp[1] # a constant
    # arriving in dImod,r/dImod = (dIexp/dImod) * coeff_exp[1]
    # but since Iexp and Imod vary linearly according to a putative (hidden) parameter k
    # (i.e. the difference I_exp[2] - I_exp[1] = I_exp[1] - I_exp[0], etc, and the same for I_model)
    # then dIexp/dImod = Delta Iexp / Delta Imod = (I_exp[1] - I_exp[0]) / (I_model[1] - I_model[0])
    if rescaleCurr:
        coeff_model = (None,((I_exp[1] - I_exp[0]) / (I_model[1] - I_model[0])) * coeff_exp[1]) # setting the angular coefficient of the I_model_rescaled
        _,coeff_model = misc.linearTransf(I_model[k:],[0,1],returnCoeff=True,coeff=coeff_model) # calculating the linear coefficient of the I_model_rescaled
        I_model_rescaled = misc.linearTransf(I_model,coeff=coeff_model)
    else:
        I_model = I_model*1.0e3  # from nA to pA
        I_model_rescaled = I_model
        coeff_model = (0.0,1.0)


    """
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ###
    ###
    ###  plotting figures
    ###
    ###
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    """

    print('   ... plotting')
    color_list = p.get_default_colors()

    exp_color = color_list[3] if (type(spkn) is int) else p.get_cold_colors
    model_color = color_list[0] if (type(spkn) is int) else p.get_hot_colors


    if showVoltage:
        factors = misc.get_number_factors(len(stimParam['I0']))
        nrows = numpy.min(factors)
        ncols = numpy.max(factors)
        figsize = ( numpy.min((float(ncols)*3.0,15.0)), numpy.min((float(nrows)*1.6,7.5)) )
        fig,ax = p.plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
        ax = ax.flatten()
        for k,I0 in enumerate(stimParam['I0']):
            colors = p.plt.get_cmap('plasma')(numpy.linspace(0,1,len(V[k])))
            for i,VV in enumerate(V[k]):
                ax[k].plot(t,VV,'-',label='sim %d'%(i+1),c=colors[i])
            ax[k].set_title('I={:g}'.format(I0))
        ax[ncols-1].legend(bbox_to_anchor=(1,1),loc='upper left')
        if saveFig:
            output.save_fig(p.plt,outFileNamePrefix_fig,'th0_vs_I_voltage',resolution=outFigResolution,file_format=outFileFormat)

    #p.plt.plot(misc.linearTransf(Iall_exp,coeff=coeff_exp),thall_exp,'s',markerfacecolor='w',markersize=3,color=color_list[2])

    fig = p.plt.figure()
    ax=p.plt.gca()
    p.errorfill(I_exp_rescaled,th_exp[:,spkn],therr_exp[:,spkn],fmt=':s', label='Experiment; I=[%g,%g]pA'%(numpy.min(I_exp),numpy.max(I_exp)),color=exp_color,ax=ax)
    p.errorfill(I_model_rescaled,th_model[:,spkn],thstd_model[:,spkn],fmt='o--', color=model_color,ax=ax,label=paramset + '; I=[%g,%g]pA'%(numpy.min(I_model),numpy.max(I_model)) )
    ax.legend()
    if rescaleCurr:
        ax.set_xlabel('Rescaled injected current, a.u.')
    else:
        ax.set_xlabel('Input current (pA)')
    ax.set_ylabel('Threshold (mV) of spike %s' % spkn)

    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix_fig,'th0_vs_I',resolution=outFigResolution,file_format=outFileFormat)
    if saveData:
        output.save_data(dict(I=I_model,linearTransfCoeff=coeff_model,spkn=numpy.arange(th_model.shape[1])[spkn],theta=th_model[:,spkn].flatten(),theta_std=thstd_model[:,spkn].flatten(),th=th_model,th_std=thstd_model,Irescaled=I_model_rescaled,I_resc_func='Ir=c1+c2*I',stimArgs=stimParam,neuronArgs=misc.remove_key(simParam,'noiseSignal')),outFileNamePrefix_data,file_suffix='th0_vs_I',file_format='npz',numpy=numpy)



    exp_color = color_list[3] if (type(spkn) is int) else p.get_cold_colors(2)[-1]
    model_color = color_list[0] if (type(spkn) is int) else p.get_hot_colors(2)[-1]

    fig = p.plt.figure()
    ax = p.plt.gca()
    p.errorfill(I_mc_for_lat,t0spk_mc,t0spk_err_mc,fmt=':s', label='Experiment; I=[%g,%g]pA'%(numpy.min(I_exp),numpy.max(I_exp)),color=exp_color,ax=ax)
    p.errorfill(I_model_rescaled,tspk_lat_avg,tspk_lat_std,fmt='o--', color=model_color,ax=ax,label=paramset + '; I=[%g,%g]pA'%(numpy.min(I_model),numpy.max(I_model)) )
    ax.legend()
    ax.set_xlabel('Input current (pA)')
    ax.set_ylabel('Latency 1st spike (ms)')

    if saveFig:
        output.save_fig(p.plt,outFileNamePrefix_fig,'latency_vs_I',resolution=outFigResolution,file_format=outFileFormat)
    if saveData:
        output.save_data(dict(I=I_model,tspk=tspk_lat_avg,tspk_std=tspk_lat_std,Irescaled=I_model_rescaled,stimArgs=stimParam,neuronArgs=misc.remove_key(simParam,'noiseSignal')),outFileNamePrefix_data,file_suffix='latency_vs_I',file_format='npz',numpy=numpy)

    p.plt.show()

    print('oe')

if __name__ == '__main__':
    main()