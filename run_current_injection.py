import os
import math
import copy
import operator
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

parser = argparse.ArgumentParser(description='Run current injection experiment on neurons')
parser = inp.add_neuron_parameters(parser)
#parser = inp.add_neuronparamset_parameters(parser)
#parser = inp.add_stimulus_parameters(parser,tStim=[250.0],DeltaTStim=[4500.0],I0=[50.0])
parser = inp.add_stimulus_parameters(parser,tStim=[250.0],DeltaTStim=[4500.0],I0=['nan'])
#parser = inp.add_stimulus_parameters(parser,tStim=[250.0],DeltaTStim=[4500.0])
parser = inp.add_output_parameters(parser,out=['currinj_spk_feat'])
parser = inp.add_simulation_parameters(parser,T=[5000.0],dt=[0.2],ntrials=[10])
parser.add_argument('-show', required=False, action='store_true', default=False, help='show voltage plots')
parser.add_argument('-savefig', required=False, action='store_true', default=False, help='show voltage plots')
parser.add_argument('-savevoltage', required=False, action='store_true', default=False, help='save voltage traces')

args = parser.parse_args()
saveData = args.save
saveVoltage = args.savevoltage
saveFig = args.savefig
showFig = args.show
outFileNamePrefix = args.out[0]
outFileFormat = args.format[0]
outFigResolution = args.dpi[0]

paramsetList = args.paramset
#paramsetList = ['EIFDTBoundSigKLR']
#showFig = True
#paramsetList = ['LIFDTBoundKLR','LIFDTK']
#saveVoltage = True
#print(paramsetList)
#paramsetList = ['LIFDTK','LIFDTmod3']

get_items = lambda mylist,*idx:list(operator.itemgetter(*idx)(mylist)) # returns the elements identified by idx from mylist

outFileNamePrefix_fig,outFileNamePrefix_data = output.create_fig_data_dir(outFileNamePrefix,'currinj','','',saveFig,saveData or saveVoltage)

if len(paramsetList) == 1:
    if paramsetList[0].lower() == 'none':
        raise ValueError('you must specify paramset')

stimArgsList = inp.get_input_stimulus_args(args=args,defaultI0='nan',returnAsListOfDict=True)
#stimArgsList = inp.get_input_stimulus_args(args=args,defaultI0=[50.0,60.0],returnAsListOfDict=True)

if not(type(stimArgsList) is list):
    stimArgsList = [stimArgsList]

if not(type(paramsetList) is list):
    paramsetList = [paramsetList]

if len(stimArgsList) != len(paramsetList):
    N = numpy.max((len(stimArgsList),len(paramsetList)))
    paramsetList = misc.repeat_to_complete(paramsetList,N,copydata=True)
    stimArgsList = misc.repeat_to_complete(stimArgsList,N,copydata=True)

N = len(paramsetList)

dt = args.dt[0]
T = args.T[0]
ntrials = args.ntrials[0]
#ntrials = 3

#print(stimArgsList)

for i,(paramset,stimParam) in enumerate(zip(paramsetList,stimArgsList)):
    neuronArgs = neu.GetDefaultParamSet(paramset,neuronArgs=inp.get_input_neuron_args(args),modified_paramsetArgs=inp.get_cmd_line_neuron_args(args))#inp.get_modified_paramset_args(paramset,args))
    simParam = dict(**neuronArgs,dt=dt,noiseSignal=lambda:0.0,ntrials=ntrials)
    if type(stimParam['I0']) is str:
        if stimParam['I0'] == 'nan':
            stimParam['I0'] = neu.GetDefaultCurrentRange(paramset)
        else:
            stimParam['I0'] = misc.get_range_from_string(stimParam['I0'])

    print('... {:s}: {:d} trials, I0 = {}'.format(simParam['neuronType'],ntrials,stimParam['I0']))
    t,I,V,th,g1,g2 = neu.RunSingleNeuron(T,stimParam,**simParam)

    print('   ... probing threshold decay')
    paramThDecay = copy.deepcopy(simParam)
    paramThDecay['I0'] = neu.GetDefaultParamSet(paramset)['I0']
    DeltaT,th_amp,th_std,_ = neu.ProbeThresholdDecay(200.0,50.0,20,**paramThDecay)
    tauTheta = misc.fit_tau_theta(DeltaT,th_amp,th_std,simParam)

    print('   ... calculating spike features')
    ISIstruct,AHPMinStruct,SpkThreshStruct,AHPAmpStruct,DeltaThetaStruct,SpkThreshDiffStruct = misc.calc_spike_features(t,I,V,th,stimParam['I0'],simParam,tauTheta=tauTheta,Vth=59.0)

    theta0 = misc.calc_threshold_max_curvature(V,threshold=20.0)
    #get_theta0 = lambda th: th[:,0] if th.shape[1] > 0 else numpy.nan
    if misc.is_list_of_list(theta0,internal_comparison_func=all):
        theta0 = [ numpy.nanmean(misc.asarray_nanfill(tthh)[:,0]) for tthh in theta0 ]
    else:
        theta0 = numpy.nanmean(misc.asarray_nanfill(theta0)[:,0])

    if showFig:
        if type(stimParam['I0']) is float:
            p.plot_complete_voltage_trace(simArgs=simParam,t=t,I=[I],V=[V],th=[th],g1=[g1],g2=[g2])
        else:
            p.plot_complete_voltage_trace(simArgs=simParam,t=t,I=get_items(I,0,-1),V=get_items(V,0,-1),th=get_items(th,0,-1),g1=get_items(g1,0,-1),g2=get_items(g2,0,-1))
        if saveFig: output.save_fig(p.plt,outFileNamePrefix_fig,resolution=outFigResolution,file_format=outFileFormat,file_suffix=paramset)
    if saveVoltage:
        if misc.is_list_of_list(V,internal_comparison_func=all):
            VV = numpy.transpose(numpy.asarray(V),axes=(1,2,0)) # VV[k] is a matrix with time as rows and current as cols
            # the following insertions are necessary to keep the same standard of Anh-Tuan experimental MAT-files
            VV = numpy.insert(VV,0,t,axis=2) # inserting the time column for all trials
            VV = numpy.insert(VV,0,[numpy.nan]+list(stimParam['I0']),axis=1) # inserting the currents row on top of each VV[k]
        else:
            # the following insertions are necessary to keep the same standard of Anh-Tuan experimental MAT-files
            VV = [ numpy.insert(v.reshape((len(t),1)),0,t,axis=1) for v in V ] # inserting the time column for all trials
            VV = [ numpy.insert(v,0,[numpy.nan]+[stimParam['I0']],axis=0) for v in VV ] # inserting the currents row on top of each VV[k]
        # the file name is to mimic Anh-Tuan standard
        output.save_data(misc.list_to_dict(VV,key_prefix='Cell1_'+paramset+'_R'),outFileNamePrefix_data,file_suffix=paramset+'_volt',file_format='mat')
    if saveData:
        #output.save_data({'t':t,'I':I,'V':V,'th':th,'g1':g1,'g2':g2,'stimArgs':stimParam,'neuronArgs':simParam},outFileNamePrefix_data,file_suffix=paramset+'_vars',file_format='npz',numpy=numpy)
        output.save_data({'ISI':ISIstruct,'AHPMin':AHPMinStruct,'SpkTh':SpkThreshStruct,'AHPAmp':AHPAmpStruct,'DeltaTh':DeltaThetaStruct,'SpkThDiff':SpkThreshDiffStruct,'theta0':theta0,'tauTheta':tauTheta,'stimArgs':stimParam,'neuronArgs':misc.remove_key(simParam,'noiseSignal')},outFileNamePrefix_data,file_suffix=paramset+'_spk_features',file_format='npz',numpy=numpy)

if showFig:
    p.plt.show()