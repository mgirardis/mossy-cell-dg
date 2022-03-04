import argparse
import numpy
import copy
from numpy.core.fromnumeric import squeeze
import wx
import modules.misc as misc
import modules.neurons as neu # import neuronArgs_neuronType,neuronArgs_paramset,neuronArgs_default,GetDefaultParamSet
import xml.etree.ElementTree as ET
import scipy.io #import loadmat
import sys
#import numpy

numpy.seterr(all='ignore')

def get_files_GUI(message='Select file...',path='',wildcard='*.npz',multiple=True,max_num_files=3):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    if multiple:
        style = style | wx.FD_MULTIPLE
    dialog = wx.FileDialog(None, message, wildcard=wildcard, style=style, defaultDir=path)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPaths()
        if multiple:
            if len(path) > max_num_files:
                path = path[:max_num_files]
        else:
            path = path[0]
    else:
        path = None
    dialog.Destroy()
    return path

def add_simulation_parameters(parser,**defaulValues):
    parser.add_argument('-ntrials', nargs=1,required=False,metavar='VALUE',type=int,default=get_param_value('ntrials',defaulValues,[10]),help='number of trials for the simulation of 1 neuron')
    parser.add_argument('-dt', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('dt',defaulValues,[0.2]),help='(ms) Time step')
    parser.add_argument('-T', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('T',defaulValues,[6000.0]),help='(ms) Total time')
    return parser

def add_neuronparamset_parameters(parser,**defaultValues):
    # simulation parameters
    parser.add_argument('-paramset', nargs='+', required=False, metavar='PARAM_SET', type=str, default=get_param_value('paramset',defaultValues,['none']), choices=['none']+neu.neuronArgs_paramset, help='standard param set for each of the models... if none is specified, then use input parameters to this script')
    parser.add_argument('-neuron', nargs=1, required=False, metavar='NEURON_TYPE', type=str, default=get_param_value('neuron',defaultValues,['LIFDT']), choices=neu.neuronArgs_neuronType, help='type of neuron to integrate')
    parser.add_argument('-noise', nargs=1, required=False, metavar='NOISE_TYPE', type=str, default=get_param_value('noise',defaultValues,['white']), choices=['white'], help='type of membrane noise (only white available)')
    parser.add_argument('-synnoise_type', nargs=1, required=False, metavar='SYNNOISE_TYPE', type=str, default=get_param_value('synnoise_type',defaultValues,['none']), choices=['none','poisson','synpoisson'], help='type of input noise')
    parser.add_argument('-noise_stddev', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('noise_stddev',defaultValues,[1.0]),help='(mV) std deviation of noise (if 0, no noise is present); D = stddev^2')
    parser.add_argument('-noise_rate', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('noise_rate',defaultValues,[0.001]),help='(Hz) rate of Poisson process (if chosen)')
    parser.add_argument('-noise_tau1', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('noise_tau1',defaultValues,[2.0]),help='(ms) rise time constant for synpoisson')
    parser.add_argument('-noise_tau2', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('noise_tau2',defaultValues,[200.0]),help='(ms) decay time constant for synpoisson')
    parser.add_argument('-noise_J', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('noise_J',defaultValues,[0.1]),help='(nA) synpoisson noise intensity')
    return parser

def add_neuron_parameters(parser,**defaultValues):
    parser = add_neuronparamset_parameters(parser,**defaultValues)

    # initial conditions
    parser.add_argument('-VInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('VInit',defaultValues,[-75.0]),help='(mV) initial membrane potential')
    parser.add_argument('-thetaInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('thetaInit',defaultValues,[-40.0]),help='(mV) initial threshold (or simply the threshold if LIF is selected)')
    parser.add_argument('-gInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('gInit',defaultValues,[0.0]),help='(nA) initial condition on K current')
    parser.add_argument('-g2Init', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('g2Init',defaultValues,[0.0]),help='(nA) initial condition on second current')
    parser.add_argument('-wInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('wInit',defaultValues,[0.0]),help='(nA) initial value of adapt current in AdEx')

    # general LIF parameters
    parser.add_argument('-VR', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('VR',defaultValues,[-45.0]),help='(mV) reset potential')
    parser.add_argument('-VB', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('VB',defaultValues,[-75.0]),help='(mV) baseline (rest) potential')
    parser.add_argument('-Vc', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('Vc',defaultValues,[-75.0]),help='(mV) second baseline (rest) potential (for QIF models) or Na activation potential (for iEIF models); closing potential for theta vs. V')
    parser.add_argument('-Vo', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('Vo',defaultValues,[-75.0]),help='(mV) second baseline (rest) potential (for QIF models) or Na activation potential (for iEIF models); opening potential for theta vs. V')
    parser.add_argument('-tau', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tau',defaultValues,[10.0]),help='(ms) membrane time constant')
    parser.add_argument('-R', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('R',defaultValues,[1.0]),help='(MOhm) membrane resistance -- from experiments: range from 100 to 200 MOhm')
    parser.add_argument('-theta0', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('theta0',defaultValues,[-40.0]),help='(mV) resting threshold  (-57.2 mV -- fitted from MATLAB) when theta0 linearly depends on Iext')

    # LIFDT parameters
    parser.add_argument('-tauTheta', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauTheta',defaultValues,[600.0]),help='(ms) threshold time constant')
    parser.add_argument('-DeltaTheta', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('DeltaTheta',defaultValues,[3.0]),help='(mV) theshold increase after spike')

    # adaptive K current parameters
    parser.add_argument('-EK', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('EK',defaultValues,[-85.0]),help='(mV) K resting potential')
    parser.add_argument('-DeltaGK', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('DeltaGK',defaultValues,[1.0]),help='(nA) increase in K current on spike')
    parser.add_argument('-tauK', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauK',defaultValues,[40.0]),help='(ms) time constant of K current')
    
    # generalized LIFDT + K current
    parser.add_argument('-thetaMax', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('thetaMax',defaultValues,[-30.0]),help='(mV) upper limit for threshold growth (if bounded model chosen)')
    parser.add_argument('-gMax', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('gMax',defaultValues,[1.0]),help='(nA) maximum value for K current growth')
    parser.add_argument('-E2', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('E2',defaultValues,[-55.0]),help='(mV) second current resting potential')
    parser.add_argument('-DeltaG2', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('DeltaG2',defaultValues,[1.0]),help='(nA) increase in second current on spike')
    parser.add_argument('-G2', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('G2',defaultValues,[1.0]),help='(nA) amplitude of slow current')
    parser.add_argument('-tau2', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tau2',defaultValues,[20.0]),help='(ms) time constant of second current')
    parser.add_argument('-rv', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('rv',defaultValues,[0.8]),help='slope of the reset potential following a spike as a function of V before the spike')
    parser.add_argument('-deltaV', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('deltaV',defaultValues,[12.0]),help='(mV) decrement in V following a spike')
    parser.add_argument('-aThetaV', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('aThetaV',defaultValues,[1.0]),help='(1/ms) proportionality constant of theta vs. V')
    parser.add_argument('-tauThetaV', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauThetaV',defaultValues,[600.0]),help='(ms) theta vs. V decay time constant')
    parser.add_argument('-Rth', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('Rth',defaultValues,[95.0]),help='MOhm (proportionality constant between theta0 and external current)')
    parser.add_argument('-mIAInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('mIAInit',defaultValues,[0.0]),help='(adim) the IA current starts closed')
    parser.add_argument('-gIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('gIA',defaultValues,[0.001]),help='(mS) max conductance of the IA current')
    parser.add_argument('-tauIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauIA',defaultValues,[100.0]),help='(ms) time constant of the IA current')
    parser.add_argument('-VIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('VIA',defaultValues,[-55.0]),help='(mV) opening potential of the IA current')
    parser.add_argument('-kIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('kIA',defaultValues,[1.0]),help='(mV) slope of the IA opening sigmoid')
    #parser.add_argument('-tauDT', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauDT',defaultValues,[10.0]),help='(ms) time constant of the DeltaTheta increment')

    # exponential LIF (adaptive) parameters
    parser.add_argument('-DeltaT', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('DeltaT',defaultValues,[1.0]),help='(adim) sharpness of spikes in LIFex or AdEx models')
    parser.add_argument('-DeltaW', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('DeltaW',defaultValues,[1.0]),help='(nA) adaptation current increase after spike for AdEx')
    parser.add_argument('-gW', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('gW',defaultValues,[1.0]),help='(nA) AdEx adapt current conductance')
    parser.add_argument('-tauW', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauW',defaultValues,[1.0]),help='(ms) AdEx adapt current time constant')

    # iEIF parameters (electric fish)
    parser.add_argument('-hsInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('hsInit', defaultValues,[ 0.21]),help='mS iEIF (slow Na inactivation IC)')
    parser.add_argument('-hfInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('hfInit', defaultValues,[ 0.21]),help='mS iEIF (fast Na inactivation IC)')
    parser.add_argument('-ki', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('ki', defaultValues,[  6.0]),help='mV iEIF (Na inactivation slope)')
    parser.add_argument('-Vi', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('Vi', defaultValues,[-63.0]),help='mV iEIF (Na inactivation reversal potential)')
    parser.add_argument('-gNa', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('gNa', defaultValues,[0.036]),help='mS iEIF (Na conductance)')
    parser.add_argument('-ENa', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('ENa', defaultValues,[50.0 ]),help='mV iEIF (Na reversal potential)')
    parser.add_argument('-tauf', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauf', defaultValues,[15.0 ]),help='ms iEIF (fast inactivation time scale)')
    parser.add_argument('-taus', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('taus', defaultValues,[500.0]),help='ms iEIF (slow inactivation time scale)')
    parser.add_argument('-ka', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('ka', defaultValues,[4.0  ]),help='mV iEIF (Na+ activation slope factor)')

    # EIF + sigmoid threshold
    parser.add_argument('-VT', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('VT', defaultValues,[10.0 ]),help='mv; central ("mean") threshold of EIF (only for the EIFDTBoundSigKLR)')
    parser.add_argument('-lf_s', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('lf_s', defaultValues,[13.43]),help='mv (logistic function slope, fitted from experiments in MC for theta vs. Iext)')
    parser.add_argument('-lf_I0', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('lf_I0', defaultValues,[0.11 ]),help='nA (rest current hold for threshold, fitted from experiments in MC for theta vs. Iext)')
    parser.add_argument('-thetasInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('thetasInit', defaultValues,[0.0  ]),help='mv (spike component of the threshold; only for EIFDTBoundSigKLR)')
    parser.add_argument('-tauRiseTheta', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauRiseTheta', defaultValues,[10.0  ]),help='ms (initial rise time scale for the threshold; only for EIFDTBoundSigKLR)')
    parser.add_argument('-Vpeak', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('Vpeak', defaultValues,[20.0 ]),help='mV; peak potential (max spike threshold) for exp IF neurons')

    # IA current from Harkin draft paper
    parser.add_argument('-hIAInit', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('hIAInit', defaultValues,[ 0.0    ]),help=' IA inactivation current initial condition (from Harkin paper draft)')
    parser.add_argument('-AmIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('AmIA', defaultValues,[ 1.61   ]),help=' activation amplitude IA current -- fitted for serotonergic neurons (from Harkin paper draft)')
    parser.add_argument('-kmIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('kmIA', defaultValues,[ 0.0985 ]),help=' 1/mV, activation slope IA current -- fitted for serotonergic neurons (from Harkin paper draft)')
    parser.add_argument('-VmIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('VmIA', defaultValues,[ -23.7  ]),help=' mV, activation half potential IA current -- fitted for serotonergic neurons (from Harkin paper draft)')
    parser.add_argument('-AhIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('AhIA', defaultValues,[ 1.03   ]),help=' inactivation amplitude IA current -- fitted for serotonergic neurons (from Harkin paper draft)')
    parser.add_argument('-khIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('khIA', defaultValues,[ -0.165 ]),help=' inactivation slope IA current -- fitted for serotonergic neurons (from Harkin paper draft)')
    parser.add_argument('-VhIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('VhIA', defaultValues,[ -59.2  ]),help=' inactivation half potential IA current -- fitted for serotonergic neurons (from Harkin paper draft)')
    parser.add_argument('-tauhIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tauhIA', defaultValues,[ 43.0   ]),help='ms inactivation timescale IA current -- fitted for serotonergic neurons (from Harkin paper draft; activation was about 7 ms)')
    parser.add_argument('-EKIA', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('EKIA', defaultValues,[ -85.0  ]),help=' mV reversal K potential for IA current -- fitted for serotonergic neurons (from Harkin paper draft)')
    
    return parser

def add_stimulus_parameters(parser,**defaultValues):
    parser.add_argument('-stimtype', nargs=1, required=False, metavar='PARAM_SET', type=str, default=get_param_value('stimtype',defaultValues,['Step']), choices=['None', 'Step', 'Ramp'], help='type of input current used to stimulate neuron')
    parser.add_argument('-I0', nargs='+',required=False,metavar='VALUE',action=InputCurrentArgAction,type=input_current_type,default=get_param_value('I0',defaultValues,['nan']),help='(nA) input current; number, many numbers, or range formatted as start:end:number')
    parser.add_argument('-tStim', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('tStim',defaultValues,[500.0]),help='(ms) time to begin external stimulation')
    parser.add_argument('-DeltaTStim', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('DeltaTStim',defaultValues,[2000.0]),help='(ms) external stimulus duration')
    parser.add_argument('-thDecayDeltaTStim', nargs=1,required=False,metavar='VALUE',type=float,default=get_param_value('thDecayDeltaTStim',defaultValues,[200.0]),help='(ms) duration of the threshold decay stimulus')
    return parser

def add_output_parameters(parser,**defaultValues):
    parser.add_argument('-thresholdmethod', nargs=1, required=False, metavar='METHOD', type=str, default=get_param_value('thresholdmethod',defaultValues,['model']), choices=['model','minderivative','maxcurvature'], help='method used to calculate the threshold at every spike... model -> uses model theta variable; minderivative -> minimum derivative method; maxcurvature -> maximum curvature method')
    parser.add_argument('-save', required=False, action='store_true', default=False, help='saves figures in a subdire called fig of the config file directory')
    parser.add_argument('-out', nargs=1, required=False, metavar='FIGURE_NAME_PREFIX', type=str, default=get_param_value('out',defaultValues,['sim_file']), help='prefix of the output figure names')
    parser.add_argument('-format', nargs=1, required=False, metavar='FIGURE_FORMAT', type=str, default=get_param_value('format',defaultValues,['png']), help='format of the figures to be saved')
    parser.add_argument('-dpi', nargs=1, required=False, metavar='FIG_DPI', type=int, default=get_param_value('dpi',defaultValues,[200]), help='resolution of output figure')
    return parser

def get_param_value(paramName,args,default):
    if paramName in args.keys():
        return args[paramName]
    return default

def input_avgrange_type(txt):
    if type(txt) is str:
        if not(':' in txt.lower()):
            txt = float(txt)
        return txt
    else:
        return float(txt)

def input_current_type(txt):
    if txt.lower() == 'nan':
        return txt
    if misc.is_valid_input_range(txt):
        return txt
    else:
        try:
            return float(txt)
        except:
            argparse.ArgumentTypeError('%s is not a valid float'%txt)
        argparse.ArgumentTypeError('%s is not a valid range string, it must be A1:A2:N'%txt)

def get_input_current(I0,stimType,neuronArgs=None,defaultI0=None):
    if type(I0) is str:
        if I0.lower() == 'nan':
            if neuronArgs is None and defaultI0 is None:
                raise ValueError('you need to define I0 for stimulus')
            if neuronArgs:
                if 'I0' in neuronArgs.keys():
                    return neuronArgs['I0']
                else:
                    if stimType.lower() != 'none':
                        raise ValueError('you need to define I0 for stimulus')
                    return 0.0
            if not(type(defaultI0) is type(None)):
                return defaultI0
        else:
            return misc.get_range_from_string(I0)
    else:
        if type(I0) is list:
            return numpy.asarray(I0)
        else:
            return I0

class InputCurrentArgAction(argparse.Action):
    def is_list_of_valid_num(self,v):
        if type(v) is list:
            try:
                x = [ float(vv) for vv in v ]
                return True
            except:
                return False
        else:
            try:
                float(v)
                return True
            except:
                return False
    def __call__(self,parser,namespace,values,option_string=None):
        if self.is_list_of_valid_num(values) or misc.is_valid_range_str(values[0]):
            setattr(namespace, self.dest, values)
        else:
            raise ValueError('I0 must be a number, many numbers or a valid range string: start:end:amount')

def get_input_stimulus_args(args=None,neuronArgs=None,defaultI0=None,returnAsListOfDict=False):
    if args is None:
        stimArgs = {'stimtype': 'None', 'I0': 'nan', 'tStim': 500.0, 'DeltaTStim': 2000.0, 'thDecayDeltaTStim': 200.0}
        # stimType == 'None' or 'Step' or 'Ramp'
        stimArgs['I0'] = 40.0
        if neuronArgs and ('I0' in neuronArgs.keys()):
            stimArgs['I0'] = neuronArgs['I0']
        return stimArgs
    stimArgs = {'stimtype':args.stimtype[0],
                'tStim': args.tStim[0],
                'DeltaTStim': args.DeltaTStim[0],
                'thDecayDeltaTStim': args.thDecayDeltaTStim[0],
                'I0':0.0}
    I0 = args.I0[0] if len(args.I0) == 1 else args.I0
    if type(I0) is list:
        if returnAsListOfDict:
            n = len(I0)
            sa = misc.get_empty_list(n)
            for i,r in enumerate(I0):
                sa[i] = copy.deepcopy(stimArgs)
                sa[i]['I0'] = get_input_current(r,stimArgs['stimtype'],neuronArgs=neuronArgs,defaultI0=defaultI0)
            return sa
        else:
            stimArgs['I0'] = get_input_current(I0,stimArgs['stimtype'],neuronArgs=neuronArgs,defaultI0=defaultI0)
            return stimArgs
    else:
        stimArgs['I0'] = get_input_current(I0,stimArgs['stimtype'],neuronArgs=neuronArgs,defaultI0=defaultI0)
        return stimArgs

def get_cmd_line_neuron_args(args):
    corresp_dict = get_dict_from_arg_to_key()
    args_dict = fix_list_args(copy.deepcopy(vars(args)))
    mod_neuron_args = []
    for n in sys.argv:
        nn = remove_arg_dashes(n)
        k = fixed_key_from_args(corresp_dict,nn)
        if k in neu.neuronArgs_default:
            mod_neuron_args.append((copy.deepcopy(k),copy.deepcopy(args_dict[nn])))
    if len(mod_neuron_args) == 0:
        return None
    else:
        return dict(mod_neuron_args)

def remove_arg_dashes(n):
    if n[0] == '-':
        return remove_arg_dashes(n[1:])
    else:
        return copy.deepcopy(n)

def get_modified_paramset_args(paramset,args):
    if paramset == 'none':
        return None
    input_args_dict = fix_list_args(copy.deepcopy(vars(args)))
    default_struct = neu.GetDefaultParamSet(paramset)
    mod_args = []
    for k,v in input_args_dict.items():
        if (k in default_struct) and (v != default_struct[k]):
            mod_args.append((k,v))
    if len(mod_args) == 0:
        return None
    else:
        return dict(mod_args)

def get_modified_input_param(parser,args):
    args_struct = vars(args)
    default_struct = vars(get_default_input_values(parser))
    mod_args = []
    for k,v in args_struct.items():
        if v != default_struct[k]:
            mod_args.append((copy.deepcopy(k),copy.deepcopy(v)))
    if len(mod_args) == 0:
        return None
    else:
        return dict(mod_args)

def get_default_input_values(parser):
    return parser.parse_args([])

def get_dict_from_arg_to_key():
    s = get_input_args_to_fix()
    return { a:k for a,k in zip(s.from_args,s.to_keys) }

def fixed_key_from_args(correspondence_dict,arg_name):
    if arg_name in correspondence_dict:
        return correspondence_dict[arg_name]
    return arg_name

def get_input_args_to_fix():
    return misc.structtype(from_args=['neuron','noise','noise_stddev','VB','VR'],to_keys=['neuronType','noiseType','noiseStd','Vb','Vr'])

def fix_list_args(args):
    """
    fix arguments args dict that are given by a list and contain a single value
    """
    for k,v in args.items():
        if (not numpy.isscalar(v)) and (len(v) == 1):
            args[k] = v[0]
    return args

def get_modified_neuron_input_args(parser,args):
    keys_to_fix = get_input_args_to_fix()
    mod_args = misc.fix_dict_keys(get_modified_input_param(parser,args), keys_to_fix.from_args, keys_to_fix.to_keys)
    return fix_list_args({ k:v for k,v in mod_args.items() if k in neu.neuronArgs_default })

def get_input_neuron_args(args):
    keys_to_fix = get_input_args_to_fix()
    neuronArgs = set_default_kwargs(copy.deepcopy(vars(args)),**neu.neuronArgs_default)
    neuronArgs = misc.fix_dict_keys(neuronArgs,keys_to_fix.from_args,keys_to_fix.to_keys)
    neuronArgs.pop('I0',None)
    neuronArgs.pop('paramset',None)
    neuronArgs.pop('theta',None)
    keys_to_remove = set(neuronArgs.keys()) - set(neu.neuronArgs_default.keys())
    for k in keys_to_remove:
        neuronArgs.pop(k,None)
    return fix_list_args(neuronArgs)
    #neuronArgs = {'neuronType': args.neuron[0],
    #            'noiseType': args.noise[0],
    #            'VInit': args.VInit[0],
    #            'thetaInit':  args.thetaInit[0],
    #            'Vr': args.VR[0],
    #            'Vb': args.VB[0],
    #            'Vc': args.Vc[0],
    #            'Vo': args.Vo[0],
    #            'theta':  args.thetaInit[0],
    #            'tau':  args.tau[0],
    #            'R':  args.R[0],
    #            'DeltaT': args.DeltaT[0],
    #            'theta0':  args.theta0[0],
    #            'thetaMax':  args.thetaMax[0],
    #            'gMax':  args.gMax[0],
    #            'tauTheta':  args.tauTheta[0],
    #            'DeltaTheta':  args.DeltaTheta[0],
    #            'gInit': args.gInit[0],
    #            'EK': args.EK[0],
    #            'tauK': args.tauK[0],
    #            'DeltaGK': args.DeltaGK[0],
    #            'g2Init':  args.g2Init[0],
    #            'E2': args.E2[0],
    #            'tau2': args.tau2[0],
    #            'DeltaG2': args.DeltaG2[0],
    #            'G2': args.G2[0],
    #            'wInit': args.wInit[0],
    #            'DeltaW': args.DeltaW[0],
    #            'tauW': args.tauW[0],
    #            'gW': args.gW[0],
    #            'gNa': args.gNa[0], # mS iEIF (Na conductance)
    #            'ENa': args.ENa[0], # mV iEIF (Na reversal potential)
    #            'tauf': args.tauf[0], # ms iEIF (fast inactivation time scale)
    #            'taus': args.taus[0], # ms iEIF (slow inactivation time scale)
    #            'ka': args.ka[0], # mV iEIF (Na+ activation slope factor)
    #            'hsInit': args.hsInit[0], # mS iEIF (slow Na inactivation IC)
    #            'hfInit': args.hfInit[0], # mS iEIF (fast Na inactivation IC)
    #            'ki': args.ki[0],# mV iEIF (Na inactivation slope)
    #            'Vi': args.Vi[0], # mV iEIF (Na inactivation reversal potential)
    #            'Rth': args.Rth[0], # MOhm (proportionality constant between theta0 and external current)
    #            'rv': args.rv[0],
    #            'deltaV': args.deltaV[0],
    #            'aThetaV': args.aThetaV[0], # 1/ms (proportionality constant of theta vs. V)
    #            'tauThetaV': args.tauThetaV[0], # ms (theta vs. V decay time constant)
    #            'mIAInit': args.mIAInit[0], # the IA current starts closed
    #            'gIA': args.gIA[0], # max conductance of the IA current
    #            'tauIA': args.tauIA[0], # ms time constant of the IA current
    #            'VIA': args.VIA[0], # opening potential of the IA current
    #            'kIA': args.kIA[0], # slope of the IA opening sigmoid
    #            'VT': args.VT[0], # parameters for the sigmoid threshold growth of the EIFDTBoundSigKLR
    #            'lf_s': args.lf_s[0], # parameters for the sigmoid threshold growth of the EIFDTBoundSigKLR
    #            'lf_I0': args.lf_I0[0], # parameters for the sigmoid threshold growth of the EIFDTBoundSigKLR
    #            'thetasInit': args.thetasInit[0], # parameters for the sigmoid threshold growth of the EIFDTBoundSigKLR
    #            'tauRiseTheta': args.tauRiseTheta[0], # parameters for the sigmoid threshold growth of the EIFDTBoundSigKLR
    #            'noiseStd': args.noise_stddev[0]}#,
    #            #'I0': args.I0[0]}
    #return neuronArgs

def avg_data_matrix(y,yErr=None,parVal=None,avgType='none',axis=1,parValRange=None,errorMethod='max',returnStdErr=False,weights=None):
    """
    averages data over axis; and yErr is
    1) errorMethod == max,
        err = max(yErr) for each spike #
    2) errorMethod == mean,
        err = sd
        sd = sqrt(mean(yErr**2))  [[i.e., the mean std dev as the sqrt of mean variance]]
        if returnStdErr is True, then return 
        err = sd/sqrt(n) , where n is the amount of samples for each piece of data
    if the data was transposed,
    use axis=0 for averaging over current

    weights -> weight given to each column of y (if axis == 1) or row (if axis == 0) in doing the averages
    """
    if misc.is_vector(y):
        axis=0
    if type(yErr) is type(None):
        yErr = numpy.zeros_like(y) + numpy.nan
    avgType = avgType.lower()
    errorMethod = errorMethod.lower()
    if not(avgType in ['all','none','hilo']):
        raise ValueError('avgType must be either all, none or hilo')
    if not(errorMethod in ['max','mean']):
        raise ValueError('errorMethod must be either max or mean')
    if misc.exists(weights):
        weights = numpy.asarray(weights)
        if ((weights.ndim == 1) and (weights.size != y.shape[axis])) or ((weights.ndim == 2) and (weights.shape != y.shape)):
            raise ValueError('you must provide one weight for each %s in y'%('row' if axis==0 else 'column'))
    if avgType == 'none':
        return y,yErr,parVal
    else:
        if axis == 1:
            get_val = lambda v,k1,k2,flat=False: v[:,k1:k2] if not flat else v[k1:k2]
        elif axis == 0:
            get_val = lambda v,k1,k2,flat=False: v[k1:k2,:] if not flat else v[k1:k2]
        else:
            raise ValueError('axis not implemented')
        if misc.exists(parValRange) and not misc.exists(parVal):
            raise ValueError('parValRange can only be used when setting parVal variable')
        k1,k2 = misc.get_range_by_values(parVal,parValRange)
        if avgType == 'all':
            if misc.exists(weights):
                nanaverage = lambda xx,**kwargs: numpy.average(numpy.ma.masked_array(xx,mask=numpy.isnan(xx)),weights=get_val(weights,k1,k2,True),**kwargs)
            else:
                nanaverage = numpy.nanmean
            yy = get_val(y,k1,k2,misc.is_vector(y))
            yyErr = get_val(yErr,k1,k2,misc.is_vector(yErr))

            yy = nanaverage(yy,axis=axis)
            if errorMethod == 'max':
                yyErr = numpy.nanmax(yyErr,axis=axis)
            else:
                yyErr = numpy.sqrt(nanaverage(yyErr**2.0,axis=axis))
                if returnStdErr:
                    n = float(k2 - k1) if k2 != k1 else 1.0
                    yyErr = yyErr / numpy.sqrt(n)
            if misc.exists(weights):
                pval = numpy.average(get_val(parVal.flatten(),k1,k2,True),weights=get_val(weights,k1,k2,True)) if misc.exists(parVal) else parVal
            else:
                pval = numpy.nanmean(get_val(parVal.flatten(),k1,k2,True)) if not(type(parVal) is type(None)) else parVal
        elif avgType == 'hilo':
            if misc.exists(weights):
                nanaverage = lambda xx,weights,**kwargs: numpy.average(numpy.ma.masked_array(xx,mask=numpy.isnan(xx)),weights=weights,**kwargs)
            else:
                weights = numpy.ones(y.shape)
                nanaverage = lambda xx,weights,**kwargs: numpy.nanmean(xx,**kwargs)
            if misc.is_vector(y):
                n = int(numpy.ceil(len(y) / 2.0))
                #yy = numpy.asarray([ numpy.nanmean(y[:n]), numpy.nanmean(y[-n:]) ])
                #yyErr = numpy.asarray([ numpy.nanmax( yErr[:n] ), numpy.nanmax( yErr[-n:] ) ])
                yy = numpy.asarray([ nanaverage(y[:n],weights[:n]), nanaverage(y[-n:],weights[-n:]) ])
                if errorMethod == 'max':
                    yyErr = numpy.asarray([ numpy.nanmax( yErr[:n] ), numpy.nanmax( yErr[-n:] ) ])
                else:
                    yyErr = numpy.asarray([ numpy.sqrt(nanaverage( yErr[:n]**2.0,weights[:n] )), numpy.sqrt(nanaverage( yErr[-n:]**2.0,weights[-n:] )) ])
                pval=None
                if misc.exists(parVal):
                    #pval = numpy.asarray( [ numpy.nanmean(parVal[:n]), numpy.nanmean(parVal[-n:])  ] )
                    pval = numpy.asarray( [ nanaverage(parVal[:n],weights[:n]), nanaverage(parVal[-n:],weights[-n:])  ] )
                return yy,yyErr,pval
            n = int(numpy.ceil(y.shape[axis] / 2.0))
            if axis == 0:
                z = y.T
                zErr = yErr.T
            else:
                z = y
                zErr = yErr
            nspk = z.shape[0]
            yy = numpy.zeros((nspk,2))
            yy[:,0] = nanaverage(z[:,:n],weights[:n],axis=1)
            yy[:,1] = nanaverage(z[:,-n:],weights[-n:],axis=1)
            #yy[:,0] = numpy.nanmean(z[:,:n],axis=1)
            #yy[:,1] = numpy.nanmean(z[:,-n:],axis=1)
            if errorMethod == 'max':
                yyErr = numpy.zeros((nspk,2))
                yyErr[:,0] = numpy.nanmax(zErr[:,:n],axis=1)
                yyErr[:,1] = numpy.nanmax(zErr[:,-n:],axis=1)
            else:
                yyErr = numpy.zeros((nspk,2))
                yyErr[:,0] = numpy.sqrt(nanaverage(zErr[:,:n]**2.0,weights[:n],axis=1))
                yyErr[:,1] = numpy.sqrt(nanaverage(zErr[:,-n:]**2.0,weights[-n:],axis=1))
                if returnStdErr:
                    yyErr = yyErr / numpy.sqrt(float(n))
            if axis == 0:
                yy = yy.T
                yyErr = yyErr.T
            if misc.exists(parVal):
                #pval = numpy.asarray( [ numpy.nanmean(parVal[:n]), numpy.nanmean(parVal[-n:])  ] )
                pval = numpy.asarray( [ nanaverage(parVal[:n],weights[:n]), nanaverage(parVal[-n:],weights[-n:])  ] )
            else:
                pval = parVal
        return yy,yyErr,pval

def get_mc_experiment_var(data,varName,transpose=False):
    """
    returns the variable varName from data.
    if transpose is false,
    then rows -> spike#, cols -> current;
    if true, then rows->current, cols->spike#
    """
    current = data[varName][0,1::2]
    spk = data[varName][1:,0].flatten() - 1.0
    y = data[varName][1:,1::2]
    yErr = data[varName][1:,2::2]
    if varName ==  'hMC_SpikeThreshold_Diff':
        spk = numpy.concatenate(([0],spk))
        y = numpy.concatenate((numpy.zeros((1,y.shape[1])),y),axis=0)
        yErr = numpy.concatenate((numpy.zeros((1,yErr.shape[1])),yErr),axis=0)
    if transpose:
        y = y.T
        yErr = yErr.T
    return spk,y,yErr,current

def get_current_from_experiment_volt_traces(file_name):
    d = scipy.io.loadmat(file_name,squeeze_me=True)
    flatten_c = lambda c: [ b for a in c for b in a ]
    c,count = numpy.unique(numpy.asarray(sorted(flatten_c([ v[0,1:] for k,v in d.items() if k[:2] != '__' ]))),return_counts=True)
    return c,count

def get_current_weights_from_volt_traces(I0,file_name,convert_to_pA=True):
    """
    I0 are the simulated input currents
    file name is the file where the voltage traces are  (D:/Dropbox/p/uottawa/data/mossy_cell_experiment/2020-06-25/hMC_StepCurrent_DataJune2021.mat)
    convert_to_pA -> if true, then do I0 * 1e3

    returns a list of weights for each entry in I0 that is found in experimental data
    if a particular I0 is not found in experimental data, it is given the weight 1
    """
    w = numpy.ones(len(I0))
    c,count = get_current_from_experiment_volt_traces(file_name)
    if convert_to_pA:
        I0 = numpy.asarray(I0) * 1.0e3
    for i,I in enumerate(I0):
        k = numpy.nonzero(c == I)[0]
        if k.size == 0:
            continue
        k = k[0] # first index of c that matches I
        w[i] = float(count[k])
    return w

def import_mc_experiment_matfile(fileName,return_as_structtype=False,**loadArgs):
    loadArgs = set_default_kwargs(loadArgs,squeeze_me=True)
    d = dict((k,v) for k,v in scipy.io.loadmat(fileName,**loadArgs).items() if k[:2] != '__')
    return misc.structtype(**d) if return_as_structtype else d

def get_voltage_trace_from_data(data,k=None):
    keys_to_return = ('t','I','V','th','g1','g2')
    if type(data['ISI']) is list:
        if k is None:
            n = len(data['ISI'])
            dd = misc.get_empty_list(n)
            for i in range(n):
                dd[i] = {mykey:data[mykey][i] for mykey in keys_to_return if mykey in data}
            return dd
        else:
            return {mykey:data[mykey][k] for mykey in keys_to_return if mykey in data}
    else:
        return {mykey:data[mykey] for mykey in keys_to_return if mykey in data}

def get_model_currinj_var(data,varName,k=None,transpose=False,rescale_current=False):
    if type(data[varName]) is list:
        if k:
            y = data[varName][k]['avg']
            yErr = data[varName][k]['std']
            spk = data[varName][k]['spk']
            current = misc.linearTransf(data[varName][k]['I'],rescale_current)
        else:
            n = len(data[varName])
            spk = misc.get_empty_list(n)
            y = misc.get_empty_list(n)
            yErr = misc.get_empty_list(n)
            current = misc.get_empty_list(n)
            for i in range(n):
                y[i] = data[varName][i]['avg']
                yErr[i] = data[varName][i]['std']
                spk[i] = data[varName][i]['spk']
                current[i] = misc.linearTransf(data[varName][i]['I'],rescale_current)
    else:
        y = data[varName]['avg']
        yErr = data[varName]['std']
        spk = data[varName]['spk']
        current = misc.linearTransf(data[varName]['I'],rescale_current)
    if transpose:
        y = [yy.T for yy in y] if type(y) is list else y.T
        yErr = [yy.T for yy in yErr] if type(yErr) is list else yErr.T
    return spk,y,yErr,current

def import_model_currinj(filename,return_as_structtype=False,**loadArgs):
    loadArgs = set_default_kwargs(loadArgs,allow_pickle=True)
    if filename is None:
        raise ValueError('filename must be provided')
    if type(filename) is list:
        n = len(filename)
        d = misc.get_empty_list(n)
        for i,f in enumerate(filename):
            d[i] = import_model_currinj(f,**loadArgs)
        r = dict.fromkeys(d[0].keys())
        for k in r.keys():
            r[k] = [ copy.deepcopy(el[k]) for el in d ]
        return misc.structtype(**r) if return_as_structtype else r
    else:
        return fix_numpy_load(filename,return_as_structtype=return_as_structtype,**loadArgs)

def fix_numpy_load(fn,return_as_structtype=False,**args):
    args = set_default_kwargs(args,allow_pickle=True)
    d = numpy.load(fn,**args)
    new_data = dict.fromkeys(d.keys(),None)
    for k in d.keys():
        if (type(d[k]) is numpy.ndarray) and (d[k].size == 1) and (d[k].shape == ()):
            new_data[k] = copy.deepcopy(d[k].item())
        else:
            new_data[k] = copy.deepcopy(d[k])
    d.close()
    return misc.structtype(**new_data) if return_as_structtype else new_data

def get_sim_elements(configFile):
    tree = ET.parse(configFile)
    root = tree.getroot()    
    sim = root.attrib
    sim = convert_fields_to_float(sim,['N_cells', 'N_recf', 'dt', 'totalTime'])
    parameters = root[0].text
    space = root[1].attrib
    space = convert_fields_to_float(space,['height', 'width', 'radius'])
    traj = [t.attrib for t in root.findall('space/trajectory')]
    traj = convert_fields_to_float(traj,['vx','vy','x0','y0'])
    rfield = [r.attrib for r in root.findall('space/receptivefield')]
    rfield = convert_fields_to_float(rfield, ['id','I0','postElement','radius','x0','y0'])
    return sim, space, traj, rfield, parameters

def convert_fields_to_float(d,keys):
    if type(d) is list:
        for e in d:
            e = convert_fields_to_float(e,keys)
    else:
        for k in keys:
            if k in d:
                if not (d[k] == ''):
                    d[k] = float(d[k])
    return d

def parse_config_params(parameters):
    p = [ x.replace('=',' ').split() for x in parameters.split(' -')]
    return { fix_par_name(x[0]): convert_to_float(x[1:]) for x in p }

def fix_par_name(p):
    if len(p) == 1:
        return p
    else:
        if p[0] == '-':
            return fix_par_name(p[1:])
        else:
            return p

def convert_to_float(x):
    """ converts elements of x into floats if possible"""
    if type(x) is list:
        y = [None] * len(x)
        for i,e in enumerate(x):
            y[i] = convert_to_float(e)
        if len(y) == 1:
            y = y[0]
    else:
        try:
            y = float(x)
        except:
            y = x
    return y

def set_default_kwargs(kwargs_dict,**default_args):
    """
    kwargs_dict is the '**kwargs' argument of any function
    this function checks whether any argument in kwargs_dict has a default value give in default_args
    if yes, and the corresponding default_args key is not in kwargs_dict, then includes it there

    this is useful to avoid duplicate key errors
    """
    if kwargs_dict is None:
        kwargs_dict = {}
    for k,v in default_args.items():
        if not (k in kwargs_dict):
            kwargs_dict[k] = v
    return kwargs_dict