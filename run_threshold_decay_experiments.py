import os
import gc
import math
import argparse
import numpy
import pandas
#from scipy.io import savemat
#from scipy.io import loadmat
import scipy.signal
import quantities
import modules.output as output
import modules.neurons as neu
import modules.plot as p
import modules.input as inp
import modules.misc as misc
import sys

#sys.argv.extend(['-paramset','EIFDTBoundSigKLR','-Iprobe','0.4'])

parser = argparse.ArgumentParser(description='Plots LIF and LIFDT neurons')
parser = inp.add_output_parameters(parser,out=['thresh_decay'])
#parser.add_argument('-savedata', required=False, action='store_true', default=False, help='saves data in npz file')
#parser.add_argument('-saveexpdata', required=False, action='store_true', default=False, help='experimental data theta-I curve calculation')
#parser.add_argument('-showvoltage', required=False, action='store_true', default=False, help='shows voltage plots for each current')
parser = inp.add_neuron_parameters(parser)
parser = inp.add_simulation_parameters(parser,T=[1000.0],ntrials=[1],dt=[0.02])
#parser.add_argument('-spk', nargs='+', required=False, metavar='SPK_SLICE', type=int, default=[0], help='spk number for displaying features vs. current, if set to -1, then defaults to the 10 first spikes (other int values plot only that spike); if list o int, then create a slice with elements [s0,s1,nspks]')
parser = inp.add_stimulus_parameters(parser,I0=['nan'],tStim=[100.0],DeltaTStim=[200.0])
parser.add_argument('-Iprobe', nargs=1, required=False, metavar='MAX_VALUE', type=float, default=[0.4], help='(nA) max intensity of the probe ramp current')
parser.add_argument('-IprobeDur', nargs=1, required=False, metavar='DURATION', type=float, default=[100.0], help='(ms) duration of the probe ramp 50 to 100 ms is recommended')
parser.add_argument('-Istim', nargs=1, required=False, metavar='MAX_VALUE', type=float, default=[0.4], help='(nA) max intensity of the initial (stimulus) ramp current')
parser.add_argument('-IstimDur', nargs=1, required=False, metavar='DURATION', type=float, default=[200.0], help='(ms) duration of the initial (stimulus) ramp (200 ms was used by Anh-Tuan)')
parser.add_argument('-delay', nargs=1, required=False, metavar='DURATION', type=float, default=[50.0], help='(ms) interval between end of stimulus and beginning of 1st probe')
parser.add_argument('-nprobes', nargs=1, required=False, metavar='NUM', type=int, default=[20], help='number of probe ramps to inject (i.e., max delay = nprobes * delay)')
parser.add_argument('-showtspk', required=False, action='store_true', default=False, help='shows threshold decay vs actual spk time in addition to delay intervals')
parser.add_argument('-savedata', required=False, action='store_true', default=False, help='saves data in npz file')


args = parser.parse_args()

paramset = args.paramset[0]
saveFig = args.save
saveData = args.savedata
showtspk = args.showtspk
outFileNamePrefix = args.out[0]
outFileFormat = args.format[0]
outFigResolution = args.dpi[0]
outFileNamePrefix_fig,outFileNamePrefix_data = output.create_fig_data_dir(outFileNamePrefix + '_' + paramset,'threshdecay','','',saveFig,saveData)

ntrials = args.ntrials[0]
dt = args.dt[0]
Iprobe = args.Iprobe[0] #275.0e-3 # nA
Iprobe_duration = args.IprobeDur[0]
Iprobe_delay = args.delay[0]
Iprobe_n = args.nprobes[0]
Istim = args.Istim[0] #2.0*Iprobe # nA
Istim_duration = args.IstimDur[0]

thresholdmethod = args.thresholdmethod[0]
if (thresholdmethod.lower() == 'model') and neu.has_no_threshold(paramset):
    thresholdmethod = 'minderivative'

neuronArgs = neu.GetDefaultParamSet(paramset,neuronArgs=inp.get_input_neuron_args(args),modified_paramsetArgs=inp.get_cmd_line_neuron_args(args))
par = {**neuronArgs,
    'ntrials':ntrials,
    'dt':dt,
    'noiseSignal': lambda:0.0 }

print('... %s: %d trials' % (par['neuronType'],ntrials))
#ProbeThresholdDecay(Istim_duration,DeltaT_rec,nRecTimes,Istim=None,Iprobe=None,Iprobe_duration=None,findpeaks_args=None,**neuronArgs)
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(Istim_duration,Iprobe_delay,Iprobe_n,Istim=Istim,Iprobe=Iprobe,Iprobe_duration=Iprobe_duration,use_findpeaks=True,thresholdmethod=thresholdmethod,**par)
th_amp = th_amp - theta0

"""
loading experimental threshold decay MC data
"""
th_normal_data = pandas.read_table('D:/Dropbox/p/uottawa/data/mossy_cell_experiment/MC_RAMP_Tau.dat',sep=' ', header=None, comment='#').to_numpy()
fitFuncExp = lambda xx,a,b: a * numpy.exp(-b * xx)
fitParam_data, _ = scipy.optimize.curve_fit(fitFuncExp, th_normal_data[:,0], th_normal_data[:,1], p0=(1.0,0.0001), maxfev=100000)

fitFunc = lambda xx,a,b: a * numpy.exp(-b * xx) #lambda xx,a,b,c: a * numpy.exp(-b * xx) + c
DT_fit = DeltaTexp[numpy.logical_not(numpy.isnan(th_amp))]
th_fit = th_amp[numpy.logical_not(numpy.isnan(th_amp))]
try:
    fitParam_model, _ = scipy.optimize.curve_fit(fitFunc, DT_fit, th_fit, p0=(1.0,1.0/par['tauTheta']), maxfev=100000) # ,par['theta0']
except ValueError as err:
    if str(err) == '`ydata` must not be empty!':
        print('ERROR ::: no spikes were found for fitting decay... try increasing Iprobe?')
        exit()
    raise
except TypeError as err:
    if str(err) == 'Improper input: N=3 must not exceed M=1':
        print('ERROR ::: not enough spikes were found for fitting decay...')
        exit()
    raise
x_fit = numpy.linspace(DeltaTexp[0],DeltaTexp[-1],100)

color_list = p.get_default_colors()
color_mc = numpy.asarray((237.0/255.0,30.0/255.0,35.0/255.0,1),dtype=float)
ax = p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par,same_panel_V_theta=True,show_tspk_curve=showtspk)
p.errorfill(th_normal_data[:,0], th_normal_data[:,1], th_normal_data[:,2], fmt='o',color=color_mc, label='Exp. data',ax=ax[3],zorder=-2)
p.plt.plot(x_fit, fitFuncExp(x_fit,*fitParam_data),'-',linewidth=2,color=color_list[2], label='Exp. data fit: $\\tau_{\\theta}=%.2f$'%(1.0/fitParam_data[1]),zorder=-1)
p.plt.plot(x_fit, fitFunc(x_fit,*fitParam_model),'-',linewidth=2,color=color_list[0], label='Model fit: $\\tau_{\\theta}=%.2f$'%(1.0/fitParam_model[1]),zorder=0)

ax[3].legend()

if saveData:
    output.save_data(dict(fitParam_model=fitParam_model,fitParam_data=fitParam_data,fitFunc='a+exp(-b*x)',DeltaT=DeltaT,th_amp=th_amp,th_std=th_std,t=t,I=I,V=V,tOff=tOff,DeltaTexp=DeltaTexp,theta0=theta0,theta=theta,neuronArgs=misc.remove_key(neuronArgs,'noiseSignal')),outFileNamePrefix_data,file_suffix='thresh_decay',file_format='npz',numpy=numpy)
if saveFig:
    output.save_fig(p.plt,outFileNamePrefix_fig,'thresh_decay',resolution=outFigResolution,file_format=outFileFormat)

p.plt.show()


exit()


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### EIFDT threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 1

par = {**neu.eifdtArgs,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

#par['tauRiseTheta'] = 20.0 # ms
#par['lf_I0'] = 0.15 # nA
#par['Rth'] = 8.0 # MOhm
#par['lf_s'] = 20.0 # mV

#par['deltaV'] = 12.0 # mV

print('... %s: %d trials' % (par['neuronType'],ntrials))

Iprobe = 0.4 #275.0e-3 # nA
Istim = 0.4 #2.0*Iprobe # nA
Iprobe_duration = 50.0
                                                      #ProbeThresholdDecay(Istim_duration,DeltaT_rec,nRecTimes,Istim=None,Iprobe=None,Iprobe_duration=None,findpeaks_args=None,**neuronArgs)
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,20,Istim=Istim,Iprobe=Iprobe,Iprobe_duration=Iprobe_duration,use_findpeaks=True,thresholdmethod='minderivative',**par)

fitFunc = lambda xx,a,b,c: a * numpy.exp(-b * xx) + c
DT_fit = DeltaTexp[numpy.logical_not(numpy.isnan(th_amp))]
th_fit = th_amp[numpy.logical_not(numpy.isnan(th_amp))]
fitParam_model, _ = scipy.optimize.curve_fit(fitFunc, DT_fit, th_fit, p0=(1.0,1.0/par['tauTheta'],par['theta0']), maxfev=100000)
x_fit = numpy.linspace(DeltaTexp[0],DeltaTexp[-1],100)

color_list = p.get_default_colors()
ax = p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par,same_panel_V_theta=True)
p.errorfill(th_normal_data[:,0], th_normal_data[:,1], th_normal_data[:,2], fmt='o',color=color_list[1], label='Exp. data',ax=ax[3],zorder=-2)
p.plt.plot(x_fit, fitFuncExp(x_fit,*fitParam_data),'-',linewidth=2,color=color_list[2], label='Exp. data fit: $\\tau_{\\theta}=%.2f$, $\\theta_0=??$'%(1.0/fitParam_data[1]),zorder=-1)

#output.save_fig(p.plt,'fig_threshdecay/'+par['neuronType']+'_','thresh_decay',resolution=150,file_format='png')

p.plt.show()


exit()


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### EIFDTBoundSigKLRIA threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 1

par = {**neu.eifhdtboundsigklriaArgs,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

#par['tauRiseTheta'] = 20.0 # ms
#par['lf_I0'] = 0.15 # nA
#par['Rth'] = 8.0 # MOhm
#par['lf_s'] = 20.0 # mV

#par['deltaV'] = 12.0 # mV

print('... %s: %d trials' % (par['neuronType'],ntrials))

Iprobe = 0.8 #275.0e-3 # nA
Istim = 0.8 #2.0*Iprobe # nA
Iprobe_duration = 50.0
                                                      #ProbeThresholdDecay(Istim_duration,DeltaT_rec,nRecTimes,Istim=None,Iprobe=None,Iprobe_duration=None,findpeaks_args=None,**neuronArgs)
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,20,Istim=Istim,Iprobe=Iprobe,Iprobe_duration=Iprobe_duration,use_findpeaks=True,thresholdmethod='minderivative',**par)

fitFunc = lambda xx,a,b,c: a * numpy.exp(-b * xx) + c
DT_fit = DeltaTexp[numpy.logical_not(numpy.isnan(th_amp))]
th_fit = th_amp[numpy.logical_not(numpy.isnan(th_amp))]
fitParam_model, _ = scipy.optimize.curve_fit(fitFunc, DT_fit, th_fit, p0=(1.0,1.0/par['tauTheta'],par['theta0']), maxfev=100000)
x_fit = numpy.linspace(DeltaTexp[0],DeltaTexp[-1],100)

color_list = p.get_default_colors()
ax = p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par,same_panel_V_theta=True)
p.errorfill(th_normal_data[:,0], th_normal_data[:,1], th_normal_data[:,2], fmt='o',color=color_list[1], label='Exp. data',ax=ax[3],zorder=-2)
p.plt.plot(x_fit, fitFuncExp(x_fit,*fitParam_data),'-',linewidth=2,color=color_list[2], label='Exp. data fit: $\\tau_{\\theta}=%.2f$, $\\theta_0=??$'%(1.0/fitParam_data[1]),zorder=-1)

#output.save_fig(p.plt,'fig_threshdecay/'+par['neuronType']+'_','thresh_decay',resolution=150,file_format='png')

p.plt.show()


exit()


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### AdEIFDTBoundSigKLR threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 1

par = {**neu.adeifhdtboundsigklrArgs,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

#par['tauRiseTheta'] = 20.0 # ms
#par['lf_I0'] = 0.15 # nA
#par['Rth'] = 8.0 # MOhm
#par['lf_s'] = 20.0 # mV

#par['deltaV'] = 12.0 # mV

print('... %s: %d trials' % (par['neuronType'],ntrials))

Iprobe = 0.5 #275.0e-3 # nA
Istim = 0.6 #2.0*Iprobe # nA
Iprobe_duration = 50.0
                                                      #ProbeThresholdDecay(Istim_duration,DeltaT_rec,nRecTimes,Istim=None,Iprobe=None,Iprobe_duration=None,findpeaks_args=None,**neuronArgs)
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,20,Istim=Istim,Iprobe=Iprobe,Iprobe_duration=Iprobe_duration,use_findpeaks=True,thresholdmethod='minderivative',**par)

fitFunc = lambda xx,a,b,c: a * numpy.exp(-b * xx) + c
DT_fit = DeltaTexp[numpy.logical_not(numpy.isnan(th_amp))]
th_fit = th_amp[numpy.logical_not(numpy.isnan(th_amp))]
fitParam_model, _ = scipy.optimize.curve_fit(fitFunc, DT_fit, th_fit, p0=(1.0,1.0/par['tauTheta'],par['theta0']), maxfev=100000)
x_fit = numpy.linspace(DeltaTexp[0],DeltaTexp[-1],100)

color_list = p.get_default_colors()
ax = p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par,same_panel_V_theta=True)
p.errorfill(th_normal_data[:,0], th_normal_data[:,1], th_normal_data[:,2], fmt='o',color=color_list[1], label='Exp. data',ax=ax[3],zorder=-2)
p.plt.plot(x_fit, fitFuncExp(x_fit,*fitParam_data),'-',linewidth=2,color=color_list[2], label='Exp. data fit: $\\tau_{\\theta}=%.2f$, $\\theta_0=??$'%(1.0/fitParam_data[1]),zorder=-1)

#output.save_fig(p.plt,'fig_threshdecay/'+par['neuronType']+'_','thresh_decay',resolution=150,file_format='png')

p.plt.show()

exit()


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### EIFDTBoundSigKLR threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 10

par = {**neu.eifhdtboundsigklrArgs,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

#par['tauRiseTheta'] = 20.0 # ms
#par['lf_I0'] = 0.15 # nA
#par['Rth'] = 1.0 # MOhm
#par['lf_s'] = 20.0 # mV

#par['deltaV'] = 12.0 # mV

print('... %s: %d trials' % (par['neuronType'],ntrials))

Iprobe = 0.35 #275.0e-3 # nA
Istim = 0.4 #2.0*Iprobe # nA
Iprobe_duration = 90.0
                                                      #ProbeThresholdDecay(Istim_duration,DeltaT_rec,nRecTimes,Istim=None,Iprobe=None,Iprobe_duration=None,findpeaks_args=None,**neuronArgs)
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,20,Istim=Istim,Iprobe=Iprobe,Iprobe_duration=Iprobe_duration,use_findpeaks=True,thresholdmethod='minderivative',**par)

fitFunc = lambda xx,a,b,c: a * numpy.exp(-b * xx) + c
DT_fit = DeltaTexp[numpy.logical_not(numpy.isnan(th_amp))]
th_fit = th_amp[numpy.logical_not(numpy.isnan(th_amp))]
fitParam_model, _ = scipy.optimize.curve_fit(fitFunc, DT_fit, th_fit, p0=(1.0,1.0/par['tauTheta'],par['theta0']), maxfev=100000)
x_fit = numpy.linspace(DeltaTexp[0],DeltaTexp[-1],100)

color_list = p.get_default_colors()
color_mc = numpy.asarray((237.0/255.0,30.0/255.0,35.0/255.0,1),dtype=float)
ax = p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par,same_panel_V_theta=True,show_tspk_curve=False,figsize=(8,8))
p.errorfill(th_normal_data[:,0], th_normal_data[:,1], th_normal_data[:,2], fmt='o',color=color_mc, label='hMC (N=13)',ax=ax[3],zorder=-2)
p.plt.plot(x_fit, fitFuncExp(x_fit,*fitParam_data),'-',linewidth=2,color=color_list[2], label='hMC fit: $\\tau_{\\theta}=%.2f$'%(1.0/fitParam_data[1]),zorder=-1)

ax[0].set_ylabel('Membrane voltage V(t), mV',fontsize=14)
ax[2].set_xlabel('Time (ms)',fontsize=14)
ax[2].set_ylabel('I(t), nA',fontsize=14)
ax[3].set_xlabel('Delay RAMP and probe (ms)',fontsize=14)
ax[3].set_ylabel('Delta threshold (mV)',fontsize=14)

lh = ax[3].get_lines()
lh[0].set_color('k')
lh[0]._label = 'Model simulation'
lh[0].set_markerfacecolor('w')
ax[3].legend()

p.plt.savefig('fig_threshdecay/'+par['neuronType']+'_thresh_decay.png',dpi=300,format='png',bbox_inches='tight')

p.plt.show()

exit()

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDTBoundKLR threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 1

par = {**neu.lifdtboundklrArgs,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

print('... %s: %d trials' % (par['neuronType'],ntrials))

par['DeltaTheta'] = 3.0

Iprobe = 0.5 #275.0e-3 # nA
Istim = 0.35 #2.0*Iprobe # nA
Iprobe_duration = 50.0


DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,20,Istim=Istim,Iprobe=Iprobe,Iprobe_duration=Iprobe_duration,**par)


fitFunc = lambda xx,a,b,c: a * numpy.exp(-b * xx) + c
DT_fit = DeltaTexp[numpy.logical_not(numpy.isnan(th_amp))]
th_fit = th_amp[numpy.logical_not(numpy.isnan(th_amp))]
fitParam_model, _ = scipy.optimize.curve_fit(fitFunc, DT_fit, th_fit, p0=(1.0,1.0/par['tauTheta'],par['theta0']), maxfev=100000)
x_fit = numpy.linspace(DeltaTexp[0],DeltaTexp[-1],100)

color_list = p.get_default_colors()
ax = p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par,same_panel_V_theta=True)
p.errorfill(th_normal_data[:,0], th_normal_data[:,1], th_normal_data[:,2], fmt='o',color=color_list[1], label='Exp. data',ax=ax[3],zorder=-2)
p.plt.plot(x_fit, fitFuncExp(x_fit,*fitParam_data),'-',linewidth=2,color=color_list[2], label='Exp. data fit: $\\tau_{\\theta}=%.2f$, $\\theta_0=??$'%(1.0/fitParam_data[1]),zorder=-1)

#output.save_fig(p.plt,'fig_threshdecay/'+par['neuronType']+'_','thresh_decay',resolution=150,file_format='png')

p.plt.show()

exit()

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDT threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 1

par = {**neu.lifdtArgs_mod3,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

print('... %s: %d trials' % (par['neuronType'],ntrials))

Iprobe_duration = 30.0
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,10,Istim=200.0,Iprobe=92.0,Iprobe_duration=Iprobe_duration,**par)
p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par)
p.plt.show()

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDTK threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 1

par = {**neu.lifdtkArgs,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

print('... %s: %d trials' % (par['neuronType'],ntrials))

Iprobe_duration = 100.0
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,10,Istim=200.0,Iprobe=86.0,Iprobe_duration=Iprobe_duration,**par)
p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par)
p.plt.show()

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFEx threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 1

par = {**neu.lifexArgs,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

print('... %s: %d trials' % (par['neuronType'],ntrials))

Iprobe_duration = 30.0
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,10,Istim=30.0,Iprobe=7.0,Iprobe_duration=Iprobe_duration,**par)
p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par)
p.plt.show()

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFAdEx threshold decay
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

ntrials = 1

par = {**neu.lifadexArgs,
       'dt':0.02,
       'noiseSignal':lambda:0.0,
       'ntrials':ntrials}

print('... %s: %d trials' % (par['neuronType'],ntrials))

Iprobe_duration = 30.0
DeltaT,th_amp,th_std,t,I,V,tOff,DeltaTexp,theta0,theta = neu.ProbeThresholdDecay(200.0,50.0,10,Istim=160.0,Iprobe=93.0,Iprobe_duration=Iprobe_duration,**par)
p.plot_threshold_decay_experiment(t,V,theta,I,DeltaT,th_amp,th_std,tOff,DeltaTexp,Iprobe_duration,par)
p.plt.show()