# -*- coding: utf-8 -*-
import copy
import numpy
import math
import modules.special_func as specf
import scipy.optimize
import modules.misc as misc
import modules.input as inp
import warnings
from numba import jit

numpy.seterr(all='ignore')

@jit(nopython=True)
def my_exp(x):
    return math.exp(x) if x < 709.782712893384 else numpy.inf #1.797693134862273e+308 # correct overflow error

@jit(nopython=True)
def logistic_f(x):
    return x/(1.0+abs(x)) # logistic function (a smooth sigmoid) between -1 (x<0) and 1 (x>0)

@jit(nopython=True)
def logistic_fder(x):
    return 1.0/(1.0+abs(x))**2.0 # logistic function (a smooth sigmoid) between -1 (x<0) and 1 (x>0)

@jit(nopython=True)
def logistic_f_np(x):
    return x/(1.0+numpy.abs(x)) # logistic function (a smooth sigmoid) between -1 (x<0) and 1 (x>0)

@jit(nopython=True)
def logistic_fder_np(x):
    return 1.0/(1.0+numpy.abs(x))**2.0 # logistic function (a smooth sigmoid) between -1 (x<0) and 1 (x>0)

@jit(nopython=True)
def boltzmann_f(x):
    return 1.0/(1.0+math.exp(-x)) # boltzmann sigmoid function for gating variables

@jit(nopython=True)
def boltzmann_fprime(x): # derivative of boltzmann sigmoid, amplitude must be A*k, where k is the slope of x
    e = math.exp(-x)
    return e/((e+1.0)*(e+1.0))

@jit(nopython=True)
def boltzmann_f_np(x):
    return 1.0/(1.0+numpy.exp(-x)) # boltzmann sigmoid function for gating variables

@jit(nopython=True)
def boltzmann_fprime_np(x): # derivative of boltzmann sigmoid, amplitude must be A*k, where k is the slope of x
    e = numpy.exp(-x)
    return e/((e+1.0)*(e+1.0))

@jit(nopython=True)
def iEIF_hinf(V,Vi,ki):
    return 1.0 / (1.0 + math.exp((V - Vi) / ki ))

@jit(nopython=True)
def iEIF_VTfn(gL,ka,Va,gNa,ENa):
    return Va - ka*math.log((gNa/gL)*(ENa-Va)/ka)

@jit(nopython=True)
def iEIF_thetafn(VT,ka,hs,hf):
    return VT - ka*math.log(hs*hf)

def calc_real_exp_threshold(Vb,DeltaT,VT,returnRest=False,Vb_offset=20.0,VT_offset=20.0,Rgw=None,RIA=None,RIAprime=None,EKIA=None,VTprime=None):
    """
    VT -> either a real number or a callable function of V; if present, then VTprime must be defined and callable too
    Rgw -> R * gW (memb resistance times the w adaptive current conductance)
    RIA -> lambda V: R * gIA * minf(V) * hinf(V)
    EKIA -> EK for IA current
    """
    if callable(VT) and (VTprime and (not callable(VTprime))):
        raise ValueError('if VT is a function, VTprime must be its derivative and must be defined')
    if callable(VT):
        f0 = lambda V: Vb - V + DeltaT * numpy.exp( (V - VT(V))/DeltaT )
        fder0 = lambda V: -VTprime(V)*numpy.exp((V-VT(V))/DeltaT) - 1.0
    else:
        f0 = lambda V: Vb - V + DeltaT * numpy.exp( (V - VT)/DeltaT )
        fder0 = lambda V: numpy.exp((V-VT)/DeltaT) - 1.0
    if RIA:
        if (not callable(RIA)) or (not callable(RIAprime)):
            raise ValueError('RIA and RIAprime must be functions of V')
        if not EKIA:
            raise ValueError('EKIA must be set: EK for IA current')
        if not RIAprime:
            raise ValueError('RIAprime(V) is the derivative of RIA(V) and must be set')
        f1 = lambda V: f0(V) - RIA(V)*(V-EKIA)
        fder1 = lambda V: fder0(V) - RIAprime(V)*(V-EKIA) - RIA(V)
    else:
        f1 = f0
        fder1 = fder0
    if Rgw:
        f2 = lambda V: f1(V) - Rgw*(V-Vb)
        fder2 = lambda V: fder1(V) - Rgw
    else:
        f2 = f1
        fder2 = fder1
    #r = scipy.optimize.newton( f2, numpy.asarray([Vb - 5.0, VT + 20.0]), fprime=fder2 )
    if callable(VT):
        VT0 = VT(0.0)
    else:
        VT0 = VT
    r1,res1 = scipy.optimize.newton( f2, Vb - Vb_offset, fprime=fder2, full_output=True, disp=False ) # newton method
    if not res1.converged: # maybe the root is close to a point of zero derivative, so let's try the secant method
        r1,res1 = scipy.optimize.newton( f2, Vb - Vb_offset, full_output=True, disp=False )
    r2,res2 = scipy.optimize.newton( f2, VT0 + VT_offset, fprime=fder2, full_output=True, disp=False )# newton method
    if not res2.converged: # maybe the root is close to a point of zero derivative, so let's try the secant method
        r2,res2 = scipy.optimize.newton( f2, VT0 + VT_offset, full_output=True, disp=False )
    Vrest = numpy.min((r1,r2))
    Vthresh = numpy.max((r1,r2))
    resRest = res1 if r1 == Vrest else res2
    resThresh = res1 if r1 == Vthresh else res2
    if returnRest: # returns resting state
        if not resRest.converged:
            warnings.warn('resting state calc did not converge')
        return Vrest
    else: # returns threshold
        if not resThresh.converged:
            warnings.warn('threshold calc did not converge')
        return Vthresh

def has_no_threshold(paramset):
    return (('EIF' in paramset) and (not ('HDT' in paramset))) or ('Ex' in paramset) or ('ex' in paramset)
neuronArgs_neuronType_temp = ['EIFDTBoundSigVKLR','EIFDTBoundSigKLRIA','EIFDTBoundSigKLR','EIFDTBoundKLR','EIFSubDT', 'EIFDT','EIFHDTBoundSigKLRIA','AdEIFHDTBoundSigKLR','EIFHDTBoundSigKLR',
                              'LIFAdEx', 'LIFDLT', 'LIFDLTBoundKLR', 'LIFDLTBoundKLRIA', 'LIFDT', 'LIFDT2K', 'LIFDTBoundK', 'LIFDTBoundKLR', 'LIFDTK',
                              'LIFDTKBound', 'LIFDTVBoundKLR', 'LIFex', 'LIFiEIF', 'LIFiEIFsf', 'QIF', 'QIFDT']
neuronArgs_neuronType = [ 'LIF', 'LIFDTKz' ] + neuronArgs_neuronType_temp
neuronArgs_paramset = ['LIFDTKzmod1', 'LIFDTKzmod2', 'LIFDTmod1', 'LIFDTmod2', 'LIFDTmod3', 'LIFiEIFsfmod1'] + neuronArgs_neuronType_temp
#neurons_without_threshold_var = [ n for n in neuronArgs_paramset if has_no_threshold(n) ]


neuronArgs_default = {
                        'neuronType': 'LIFDTK', # or
                        'noiseType': 'white',
                        'VInit': -67.0, # mV (initial condition of V)
                        'thetaInit': -40.0, # mV (initial condition of the threshold)
                        'Vr': -45.0, # mV (reset potential)
                        'Vb': -67.0, # mV (baseline potential -- bias)
                        'Vo': 0.0, # mV (nothing, used for QIF, openning potential for the theta vs V)
                        'Vc': 0.0, # mV (nothing, used for QIF, closing potential for the theta vs V)
                        'theta':  -40.0, # mV (initial threshold, same as thetaInit)
                        'tau':  38.0, # ms (membrane time constant)
                        'R':  1.0, # MOhm (external current channel resistance)
                        'DeltaT':  1.0, # adim (sharpness of spike for LIFex or AdEx)
                        'theta0':  -40.0, # mV (resting state threshold)
                        'thetaMax': -30.0, # mV (upper limit for the growth of theta, if bounded model chosen)
                        'gMax': 1.0, # nA (max value of the K current)
                        'tauTheta':  600.0, # ms (threshold time constant)
                        'DeltaTheta':  2.5, # mV (threshold increment on spike)
                        'gInit': 0.0, # nA (initial condition on K conductance)
                        'EK': -85.0, # mV (K resting potential)
                        'tauK': 40.0, # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
                        'DeltaGK': 1.0, # nA (increase in K conductance on spike)
                        'E2': -65.0, # mV (2nd current resting potential)
                        'tau2': 20.0, # ms (2nd current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
                        'G2': 1.0, # nA (amplitude of slow current)
                        'g2Init': 0.0, # mV (initial condition of g2)
                        'DeltaG2': 1.0, # nA (increase in K conductance on spike)
                        'wInit': 0.0, # adex adapt current init cond
                        'DeltaW': 1.0, # adex adapt current increment
                        'tauW': 10.0, # adex adapt curr time const
                        'gW': 1.0, # adex adapt curr conductance
                        'gNa': 0.036, # mS iEIF (Na conductance)
                        'ENa': 50.0, # mV iEIF (Na reversal potential)
                        'tauf': 15.0, # ms iEIF (fast inactivation time scale)
                        'taus': 500.0, # ms iEIF (slow inactivation time scale)
                        'ka': 4.0, # mV iEIF (Na+ activation slope factor)
                        'hsInit': 0.21, # mS iEIF (slow Na inactivation IC)
                        'hfInit': 0.21, # mS iEIF (fast Na inactivation IC)
                        'ki': 6.0 ,# mV iEIF (Na inactivation slope)
                        'Vi': -63.0, # mV iEIF (Na inactivation reversal potential)
                        'rv': 0.8, # slope of the VR as a function of V_before_spike (value fitted for MCs)
                        'deltaV': 12.0, # mV (decrement in V following a spike, value fitted for MCs)
                        'Rth': 95.0, # MOhm (proportionality constant between theta0 and external current)
                        'aThetaV': 1.0, # 1/ms (proportionality constant of theta vs. V)
                        'tauThetaV': 600.0, # ms (theta vs. V decay time constant)
                        'mIAInit': 0.0, # the IA current starts closed
                        'tauIA': 100.0, # ms time constant of the IA current
                        'VIA': -55.0, # opening potential of the IA current
                        'kIA': 1.0, # slope of the IA opening sigmoid
                        'gIA': 0.001, # max conductance of the IA current
                        'hIAInit': 0.0, # IA inactivation current initial condition (from Harkin paper draft)
                        'AmIA': 1.61, # activation amplitude IA current -- fitted for serotonergic neurons (from Harkin paper draft)
                        'kmIA': 0.0985, # 1/mV, activation slope IA current -- fitted for serotonergic neurons (from Harkin paper draft)
                        'VmIA': -23.7, # mV, activation half potential IA current -- fitted for serotonergic neurons (from Harkin paper draft)
                        'AhIA': 1.03, # inactivation amplitude IA current -- fitted for serotonergic neurons (from Harkin paper draft)
                        'khIA': -0.165,# inactivation slope IA current -- fitted for serotonergic neurons (from Harkin paper draft)
                        'VhIA': -59.2,# inactivation half potential IA current -- fitted for serotonergic neurons (from Harkin paper draft)
                        'tauhIA': 43.0,#ms inactivation timescale IA current -- fitted for serotonergic neurons (from Harkin paper draft; activation was about 7 ms)
                        'EKIA': -85.0,# mV reversal K potential for IA current -- fitted for serotonergic neurons (from Harkin paper draft)
                        'VT': 10.0, # mv; central ("mean") threshold of EIF (only for the EIFHDTBoundSigKLR)
                        'lf_s': 13.43, # mv (logistic function slope, fitted from experiments in MC for theta vs. Iext)
                        'lf_I0': 0.11, # nA (rest current hold for threshold, fitted from experiments in MC for theta vs. Iext)
                        'tauRiseTheta':  10.0, # ms (threshold time constant for rise dynamics for EIFHDTBoundSigKLR)
                        'thetasInit': 0.0, # mv (spike component of the threshold; only for EIFHDTBoundSigKLR)
                        'Vpeak': 20.0, #mV (escape potential that generates a spike for Exp IF)
                        'noiseStd': 0.5, # mV
                        'synnoise_type': 'none',
                        'noise_rate': 0.001, # Hz (Poisson rate)
                        'noise_tau1': 2.0, # ms (rise time)
                        'noise_tau2': 200.0, # ms (decay time)
                        'noise_J': 18.0, # mV (intensity)
                        'I0': 40.0 # nA (external current)
                     }
neuronArgs_default['g2Init'] = neuronArgs_default['G2'] * (neuronArgs_default['E2'] - neuronArgs_default['Vb'])

lifdtArgs_default = neuronArgs_default.copy()
lifdtArgs_mod1 = neuronArgs_default.copy()
lifdtArgs_mod2 = neuronArgs_default.copy()
lifdtArgs_mod3 = neuronArgs_default.copy()
lifdltArgs = neuronArgs_default.copy()
lifdtkArgs = neuronArgs_default.copy()
lifdtboundkArgs = neuronArgs_default.copy()
lifdtboundklrArgs = neuronArgs_default.copy()
lifdltboundklrArgs = neuronArgs_default.copy()
lifdltboundklriaArgs = neuronArgs_default.copy()
lifdtvboundklrArgs = neuronArgs_default.copy()
lifdtkboundArgs = neuronArgs_default.copy()
lifdt2kArgs = neuronArgs_default.copy()
lifdtkzArgs_mod1 = neuronArgs_default.copy()
lifdtkzArgs_mod2 = neuronArgs_default.copy()
qifArgs = neuronArgs_default.copy()
qifdtArgs = neuronArgs_default.copy()
lifexArgs = neuronArgs_default.copy()
lifadexArgs = neuronArgs_default.copy()
lifieifsfArgs = neuronArgs_default.copy()
lifieifsfArgs_mod1 = neuronArgs_default.copy()
lifieifArgs = neuronArgs_default.copy()
eifdtArgs = neuronArgs_default.copy()
eifdtboundklrArgs = neuronArgs_default.copy()
eifdtboundsigklrArgs = neuronArgs_default.copy()
eifdtboundsigvklrArgs = neuronArgs_default.copy()
eifdtboundsigklriaArgs = neuronArgs_default.copy()
eifsubdtArgs = neuronArgs_default.copy()
eifhdtboundsigklrArgs = neuronArgs_default.copy()
eifhdtboundsigklriaArgs = neuronArgs_default.copy()
adeifhdtboundsigklrArgs = neuronArgs_default.copy()

lifdtArgs_default['neuronType'] = 'LIFDT' # or
lifdtArgs_default['VInit'] = 0.0 # mV (initial condition of V)
lifdtArgs_default['thetaInit'] = 10.0 # mV (initial condition of the threshold)
lifdtArgs_default['Vr'] = 0.0 # mV (reset potential)
lifdtArgs_default['Vb'] = 0.0 # mV (baseline potential -- bias)
lifdtArgs_default['Vc'] = 0.0 # mV (nothing, used for QIF)
lifdtArgs_default['theta'] = 10.0 # mV (initial threshold, same as thetaInit)
lifdtArgs_default['tau'] = 10.0 # ms (membrane time constant)
lifdtArgs_default['R'] = 1.0 # MOhm (external current channel resistance)
lifdtArgs_default['theta0'] = 10.0 # mV (resting state threshold)
lifdtArgs_default['tauTheta'] = 100.0 # ms (threshold time constant)
lifdtArgs_default['DeltaTheta'] = 2.0 # mV (threshold increment on spike)
lifdtArgs_default['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdtArgs_default['EK'] = 0.0 # mV (K resting potential)
lifdtArgs_default['tauK'] = 10.0 # ms (K current recovery time)
lifdtArgs_default['DeltaGK'] = 0.0 # nA (increase in K conductance on spike)
lifdtArgs_default['noiseStd'] = 1.0 # mV
lifdtArgs_default['I0'] = 29.0 # nA

lifdtArgs_mod1['neuronType'] = 'LIFDT' # or
lifdtArgs_mod1['VInit'] = -75.0 # mV (initial condition of V)
lifdtArgs_mod1['thetaInit'] =  -40.0 # mV (initial condition of the threshold)
lifdtArgs_mod1['Vr'] = -45.0 # mV (reset potential)
lifdtArgs_mod1['Vb'] = -75.0 # mV (baseline potential -- bias)
lifdtArgs_mod1['Vc'] = 0.0 # mV (nothing, used for QIF)
lifdtArgs_mod1['theta'] =  -40.0 # mV (initial threshold, same as thetaInit)
lifdtArgs_mod1['tau'] =  38.0 # ms (membrane time constant)
lifdtArgs_mod1['R'] =  1.0 # MOhm (external current channel resistance)
lifdtArgs_mod1['theta0'] =  -40.0 # mV (resting state threshold)
lifdtArgs_mod1['tauTheta'] =  546.0 # ms (threshold time constant)
lifdtArgs_mod1['DeltaTheta'] =  3.0 # mV (threshold increment on spike)
lifdtArgs_mod1['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdtArgs_mod1['EK'] = 0.0 # mV (K resting potential)
lifdtArgs_mod1['tauK'] = 10.0 # ms (K current recovery time)
lifdtArgs_mod1['DeltaGK'] = 0.0 # nA (increase in K conductance on spike)
lifdtArgs_mod1['noiseStd'] = 0.5 # mV
lifdtArgs_mod1['I0'] = 40.0 # nA (external current)

lifdtArgs_mod2['neuronType'] = 'LIFDT' # or
lifdtArgs_mod2['VInit'] = -75.0 # mV (initial condition of V)
lifdtArgs_mod2['thetaInit'] =  -40.0 # mV (initial condition of the threshold)
lifdtArgs_mod2['Vr'] = -45.0 # mV (reset potential)
lifdtArgs_mod2['Vb'] = -75.0 # mV (baseline potential -- bias)
lifdtArgs_mod2['Vc'] = 0.0 # mV (nothing, used for QIF)
lifdtArgs_mod2['theta'] =  -40.0 # mV (initial threshold, same as thetaInit)
lifdtArgs_mod2['tau'] =  38.0 # ms (membrane time constant)
lifdtArgs_mod2['R'] =  1.0 # MOhm (external current channel resistance)
lifdtArgs_mod2['theta0'] =  -40.0 # mV (resting state threshold)
lifdtArgs_mod2['tauTheta'] =  600.0 # ms (threshold time constant)
lifdtArgs_mod2['DeltaTheta'] =  2.5 # mV (threshold increment on spike)
lifdtArgs_mod2['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdtArgs_mod2['EK'] = 0.0 # mV (K resting potential)
lifdtArgs_mod2['tauK'] = 10.0 # ms (K current recovery time)
lifdtArgs_mod2['DeltaGK'] = 0.0 # nA (increase in K conductance on spike)
lifdtArgs_mod2['noiseStd'] = 0.5 # mV
lifdtArgs_mod2['I0'] = 40.0 # nA (external current)

lifdtArgs_mod3['neuronType'] = 'LIFDT' # or
lifdtArgs_mod3['VInit'] = -67.0 # mV (initial condition of V)
lifdtArgs_mod3['thetaInit'] =  -40.0 # mV (initial condition of the threshold)
lifdtArgs_mod3['Vr'] = -60.0 # mV (reset potential)
lifdtArgs_mod3['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdtArgs_mod3['Vc'] = 0.0 # mV (nothing, used for QIF)
lifdtArgs_mod3['theta'] =  -40.0 # mV (initial threshold, same as thetaInit)
lifdtArgs_mod3['tau'] =  38.0 # ms (membrane time constant)
lifdtArgs_mod3['R'] =  1.0 # MOhm (external current channel resistance)
lifdtArgs_mod3['theta0'] =  -40.0 # mV (resting state threshold)
lifdtArgs_mod3['tauTheta'] =  600.0 # ms (threshold time constant)
lifdtArgs_mod3['DeltaTheta'] =  2.0 # mV (threshold increment on spike)
lifdtArgs_mod3['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdtArgs_mod3['EK'] = 0.0 # mV (K resting potential)
lifdtArgs_mod3['tauK'] = 10.0 # ms (K current recovery time)
lifdtArgs_mod3['DeltaGK'] = 0.0 # nA (increase in K conductance on spike)
lifdtArgs_mod3['noiseStd'] = 0.5 # mV
lifdtArgs_mod3['I0'] = 40.0 # nA (external current)

lifexArgs['neuronType'] = 'LIFex' # or
lifexArgs['VInit'] = -67.0 # mV (initial condition of V)
lifexArgs['Vr'] = -50.0 # mV (reset potential)
lifexArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifexArgs['Vc'] = 0.0 # mV (nothing, used for QIF)
lifexArgs['tau'] =  38.0 # ms (membrane time constant)
lifexArgs['R'] =  150.0 # MOhm (external current channel resistance)
lifexArgs['DeltaT'] =  1.0 # adim (sharpness of spike for LIFex or AdEx)
lifexArgs['theta0'] =  -40.0 # mV (resting state threshold)
lifexArgs['tauTheta'] =  600.0 # ms (threshold time constant)
lifexArgs['DeltaTheta'] =  2.0 # mV (threshold increment on spike)
lifexArgs['gInit'] = 0.0 # nA (initial condition on K conductance)
lifexArgs['EK'] = 0.0 # mV (K resting potential)
lifexArgs['tauK'] = 10.0 # ms (K current recovery time)
lifexArgs['DeltaGK'] = 0.0 # nA (increase in K conductance on spike)
lifexArgs['noiseStd'] = 0.5 # mV
lifexArgs['I0'] = 0.2 # nA (external current)
lifexArgs['thetaInit'] =  lifexArgs['theta0'] # mV (initial condition of the threshold)
lifexArgs['theta'] =  lifexArgs['theta0'] # mV (initial threshold, same as thetaInit)
lifexArgs['Vpeak'] = calc_real_exp_threshold(lifexArgs['Vb'],lifexArgs['DeltaT'],lifexArgs['theta0'])

eifdtArgs['neuronType'] = 'EIFDT' # or
eifdtArgs['Vr'] = -52.0 # mV (reset potential)
eifdtArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
eifdtArgs['tau'] =  38.0 # ms (membrane time constant)
eifdtArgs['R'] =  150.0 # MOhm (external current channel resistance)
eifdtArgs['DeltaT'] = 0.1 # mV (sharpness of spike for LIFex or AdEx)
eifdtArgs['VT'] =  -55.0 # mV (resting state threshold)
eifdtArgs['tauTheta'] =  540.0 # ms (threshold time constant)
eifdtArgs['DeltaTheta'] =  2.0 # mV (threshold increment on spike)
eifdtArgs['noiseStd'] = 0.5 # mV
eifdtArgs['I0'] = 0.15 # nA (external current)
eifdtArgs['VInit'] = numpy.nan # mV (initial condition of V)
eifdtArgs['thetaInit'] = eifdtArgs['VT'] # mV (initial condition of the threshold)
eifdtArgs['theta0'] = eifdtArgs['VT'] # mV (initial condition of the threshold)
eifdtArgs['theta'] = eifdtArgs['VT'] # mV (initial threshold, same as thetaInit)
eifdtArgs['Vpeak'] = -20.0 # calc_real_exp_threshold(lifexArgs['Vb'],lifexArgs['DeltaT'],lifexArgs['theta0']) # 20.0 # mV

eifdtboundklrArgs = eifdtArgs.copy()
eifdtboundklrArgs['neuronType'] = 'EIFDTBoundKLR' # or
eifdtboundklrArgs['Vr'] = -52.0 # mV (reset potential)
eifdtboundklrArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
eifdtboundklrArgs['tau'] =  38.0 # ms (membrane time constant)
eifdtboundklrArgs['R'] =  150.0 # MOhm (external current channel resistance)
eifdtboundklrArgs['DeltaT'] = 0.1 # mV (sharpness of spike for LIFex or AdEx)
eifdtboundklrArgs['VT'] =  -55.0 # mV (resting state threshold)
eifdtboundklrArgs['thetaMax'] =  20.0 # mV (upper limit for growth of theta) # -20
eifdtboundklrArgs['tauTheta'] =  540.0 # ms (threshold time constant)
eifdtboundklrArgs['DeltaTheta'] =  2.0 # mV (threshold increment on spike)
eifdtboundklrArgs['EK'] = -70.0 # mV (K resting potential)
eifdtboundklrArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
eifdtboundklrArgs['DeltaGK'] = 0.5 / eifdtboundklrArgs['R'] # nA (increase in K conductance on spike) # 0.5
eifdtboundklrArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
eifdtboundklrArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
eifdtboundklrArgs['noiseStd'] = 0.5 # mV
eifdtboundklrArgs['I0'] = 0.15 # nA (external current)
eifdtboundklrArgs['VInit'] = numpy.nan # mV (initial condition of V)
eifdtboundklrArgs['thetaInit'] = eifdtboundklrArgs['VT'] # mV (initial condition of the threshold)
eifdtboundklrArgs['gInit']     = 0.0  # nA (initial condition on K conductance)
eifdtboundklrArgs['theta0']    = eifdtboundklrArgs['VT'] # mV (initial condition of the threshold)
eifdtboundklrArgs['theta']     = eifdtboundklrArgs['VT'] # mV (initial threshold, same as thetaInit)
eifdtboundklrArgs['Vpeak']     = -20.0 # calc_real_exp_threshold(lifexArgs['Vb'],lifexArgs['DeltaT'],lifexArgs['theta0']) # 20.0 # mV

eifdtboundsigklrArgs = eifdtboundklrArgs.copy()
eifdtboundsigklrArgs['neuronType'] = 'EIFDTBoundSigKLR' # or
eifdtboundsigklrArgs['Vr'] = -48.0 #-52.0 # mV (reset potential)
eifdtboundsigklrArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
eifdtboundsigklrArgs['tau'] =  38.0 # ms (membrane time constant)
eifdtboundsigklrArgs['R'] =  150.0 # MOhm (external current channel resistance)
eifdtboundsigklrArgs['DeltaT'] = 0.1 # mV (sharpness of spike for LIFex or AdEx)
#eifdtboundsigklrArgs['VT'] =  -45.0 #-55.0 # mV (resting state threshold) # VT is now a variable made of thetas + theta
eifdtboundsigklrArgs['Rth'] =  20.0#13.43 # MOhm (threshold resistance to external input, fitted from th vs. I from MC experiments) -- other value: 8.0
eifdtboundsigklrArgs['lf_s'] =  13.44#13.44 # mV (sigmoid slope, fitted from th vs. I from MC experiments) -- other value: 20.0
eifdtboundsigklrArgs['lf_I0'] =  0.12#0.11 # nA (threshold baseline current from th vs. I from MC experiments) -- other value: 0.15
eifdtboundsigklrArgs['theta0'] =  -48.5#-50.0#-46.67 # mV (resting threshold, fitted for th vs. I from MC experiments)
eifdtboundsigklrArgs['tauRiseTheta'] = 20.0#15.0 # ms # from 1 to 20 ms
eifdtboundsigklrArgs['thetaMax'] =  30.0#20.0 # mV (upper limit for growth of theta) # -20
eifdtboundsigklrArgs['tauTheta'] =  300.0#210.0#250.0#500.0 # ms (threshold time constant)
eifdtboundsigklrArgs['DeltaTheta'] =  2.0#2.5#3.0 # mV (threshold increment on spike)
eifdtboundsigklrArgs['EK'] = -70.0 # mV (K resting potential)
eifdtboundsigklrArgs['tauK'] = 30.0#40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
eifdtboundsigklrArgs['DeltaGK'] = 0.001#0.0033333333 # nA (increase in K conductance on spike) # 0.5
eifdtboundsigklrArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
eifdtboundsigklrArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
eifdtboundsigklrArgs['noiseStd'] = 0.5 # mV
eifdtboundsigklrArgs['I0'] = 0.15 # nA (external current)
eifdtboundsigklrArgs['VInit'] = numpy.nan # mV (initial condition of V)
eifdtboundsigklrArgs['thetaInit'] = numpy.nan #eifdtboundsigklrArgs['VT'] # mV (initial condition of the threshold)
eifdtboundsigklrArgs['thetasInit'] = 0.0
eifdtboundsigklrArgs['gInit']     = 0.0  # nA (initial condition on K conductance)
#eifdtboundsigklrArgs['theta0']    = eifdtboundsigklrArgs['VT'] # mV (initial condition of the threshold)
eifdtboundsigklrArgs['Vpeak']     = -20.0 # calc_real_exp_threshold(lifexArgs['Vb'],lifexArgs['DeltaT'],lifexArgs['theta0']) # 20.0 # mV
#print('thetaRest = %.2f'%(eifdtboundsigklrArgs['theta0'] + eifdtboundsigklrArgs['lf_s']*logistic_f(-eifdtboundsigklrArgs['Rth']*eifdtboundsigklrArgs['lf_I0'])))

eifdtboundsigvklrArgs = eifdtboundsigklrArgs.copy()
eifdtboundsigvklrArgs['lf_I0'] = 0.5

eifdtboundsigklriaArgs = eifdtboundsigklrArgs.copy()
eifdtboundsigklriaArgs['neuronType'] = 'EIFDTBoundSigKLRIA' # or
eifdtboundsigklriaArgs['Vr'] = -48.0 #-52.0 # mV (reset potential)
eifdtboundsigklriaArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
eifdtboundsigklriaArgs['tau'] =  38.0 # ms (membrane time constant)
eifdtboundsigklriaArgs['R'] =  150.0 # MOhm (external current channel resistance)
eifdtboundsigklriaArgs['DeltaT'] = 0.1 # mV (sharpness of spike for LIFex or AdEx)
eifdtboundsigklriaArgs['Rth'] =  8.43 # MOhm (threshold resistance to external input, fitted from th vs. I from MC experiments) -- other value: 8.0
eifdtboundsigklriaArgs['lf_s'] =  20.44 # mV (sigmoid slope, fitted from th vs. I from MC experiments) -- other value: 20.0
eifdtboundsigklriaArgs['lf_I0'] =  0.11 # nA (threshold baseline current from th vs. I from MC experiments) -- other value: 0.15
eifdtboundsigklriaArgs['theta0'] =  -50.0#-46.67 # mV (resting threshold, fitted for th vs. I from MC experiments)
eifdtboundsigklriaArgs['tauRiseTheta'] = 15.0 # ms # from 1 to 20 ms
eifdtboundsigklriaArgs['thetaMax'] =  20.0 # mV (upper limit for growth of theta) # -20
eifdtboundsigklriaArgs['tauTheta'] =  500.0 # ms (threshold time constant)
eifdtboundsigklriaArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike)
eifdtboundsigklriaArgs['EK'] = -70.0 # mV (K resting potential)
eifdtboundsigklriaArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
eifdtboundsigklriaArgs['DeltaGK'] = 0.5 / eifdtboundsigklrArgs['R'] # nA (increase in K conductance on spike) # 0.5
eifdtboundsigklriaArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
eifdtboundsigklriaArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
eifdtboundsigklriaArgs['noiseStd'] = 0.5#0.2 # mV
eifdtboundsigklriaArgs['I0'] = 0.4 # nA (external current)
eifdtboundsigklriaArgs['VInit'] = numpy.nan # mV (initial condition of V)
eifdtboundsigklriaArgs['thetaInit'] = numpy.nan #eifdtboundsigklrArgs['VT'] # mV (initial condition of the threshold)
eifdtboundsigklriaArgs['thetasInit'] = 0.0
eifdtboundsigklriaArgs['gInit']     = 0.0  # nA (initial condition on K conductance)
eifdtboundsigklriaArgs['Vpeak']     = -20.0 # calc_real_exp_threshold(lifexArgs['Vb'],lifexArgs['DeltaT'],lifexArgs['theta0']) # 20.0 # mV
eifdtboundsigklriaArgs['hIAInit'] = 0.0 # IA inactivation current initial condition (from Harkin paper draft)
eifdtboundsigklriaArgs['gIA']    = 0.01 #0.03 # max conductance of the IA current
eifdtboundsigklriaArgs['AmIA']   = 1.61 # activation amplitude IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifdtboundsigklriaArgs['kmIA']   = 0.15 #0.0985 # 1/mV, activation slope IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifdtboundsigklriaArgs['VmIA']   = -23.7 # mV, activation half potential IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifdtboundsigklriaArgs['AhIA']   = 1.03 # inactivation amplitude IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifdtboundsigklriaArgs['khIA']   = -1.0#-0.165# inactivation slope IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifdtboundsigklriaArgs['VhIA']   = -60.0# inactivation half potential IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifdtboundsigklriaArgs['tauhIA'] = 43.0#ms inactivation timescale IA current -- fitted for serotonergic neurons (from Harkin paper draft; activation was about 7 ms)
eifdtboundsigklriaArgs['EKIA']   = -60.0#-85.0# mV reversal K potential for IA current -- fitted for serotonergic neurons (from Harkin paper draft)
#print('thetaRest = %.2f'%(eifdtboundsigklrArgs['theta0'] + eifdtboundsigklrArgs['lf_s']*logistic_f(-eifdtboundsigklrArgs['Rth']*eifdtboundsigklrArgs['lf_I0'])))


eifsubdtArgs = eifdtArgs.copy()
eifsubdtArgs['neuronType'] = 'EIFSubDT' # or
eifsubdtArgs['Vr'] = -52.0 # mV (reset potential)
eifsubdtArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
eifsubdtArgs['tau'] =  38.0 # ms (membrane time constant)
eifsubdtArgs['R'] =  150.0 # MOhm (external current channel resistance)
eifsubdtArgs['G2'] =  0.2 #  (external current channel resistance)
eifsubdtArgs['DeltaT'] = 0.1 # mV (sharpness of spike for LIFex or AdEx)
eifsubdtArgs['VT'] =  -55.0 # mV (resting state threshold)
eifsubdtArgs['tauTheta'] =  540.0 # ms (threshold time constant)
eifsubdtArgs['DeltaTheta'] =  2.0 # mV (threshold increment on spike)
eifsubdtArgs['noiseStd'] = 0.5 # mV
eifsubdtArgs['I0'] = 0.2 # nA (external current)
eifsubdtArgs['VInit'] = numpy.nan # mV (initial condition of V)
eifsubdtArgs['thetaInit'] = eifsubdtArgs['VT'] # mV (initial condition of the threshold)
eifsubdtArgs['theta0'] =    eifsubdtArgs['VT'] # mV (initial condition of the threshold)
eifsubdtArgs['theta'] =     eifsubdtArgs['VT'] # mV (initial threshold, same as thetaInit)
eifsubdtArgs['Vpeak'] = -20.0 # mV

lifadexArgs['neuronType'] = 'LIFAdEx' # or
lifadexArgs['VInit'] = -67.0 # mV (initial condition of V)
lifadexArgs['Vr'] = -48.0 # mV (reset potential)
lifadexArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifadexArgs['tau'] =  38.0 # ms (membrane time constant)
lifadexArgs['R'] =  1.0 # MOhm (external current channel resistance)
lifadexArgs['theta0'] = -41.0 # mV (resting state threshold)
lifadexArgs['DeltaT'] =  1.0 # adim (sharpness of spike for LIFex or AdEx)
lifadexArgs['wInit'] = 0.0 # init cond adex
lifadexArgs['gW'] = 1.0 # adapt current conductance
lifadexArgs['DeltaW'] = 1.0 # adapt current increase
lifadexArgs['tauW'] = 100.0 # adapt current time const
lifadexArgs['noiseStd'] = 0.5 # mV
lifadexArgs['I0'] = 50.0 # nA (external current)
lifadexArgs['thetaInit'] =  lifadexArgs['theta0'] # mV (initial condition of the threshold)
lifadexArgs['theta'] =  lifadexArgs['theta0'] # mV (initial threshold, same as thetaInit)
lifadexArgs['Vpeak'] = calc_real_exp_threshold(lifadexArgs['Vb'],lifadexArgs['DeltaT'],lifadexArgs['theta0'])

lifieifsfArgs['neuronType'] = 'LIFiEIFsf' # or
lifieifsfArgs['VInit'] = -55.0 # mV (initial condition of V)
lifieifsfArgs['Vr'] = -70.0 # mV (reset potential)
lifieifsfArgs['Vb'] = -55.0 # mV (baseline potential -- bias)
lifieifsfArgs['Vc'] = -38.6 # Va of the original iEIF mV activation potential
lifieifsfArgs['ENa'] = 50.0 # mV iEIF (Na reversal potential)
lifieifsfArgs['gNa'] = 0.036 # mS iEIF (Na conductance)
lifieifsfArgs['tau'] =  5.0 # ms (membrane time constant)
lifieifsfArgs['tauf'] = 15.0 # ms iEIF (fast inactivation time scale)
lifieifsfArgs['taus'] = 500.0 # ms iEIF (slow inactivation time scale)
lifieifsfArgs['ka'] = 4.0 # mV iEIF (Na+ activation slope factor)
lifieifsfArgs['ki'] = 6.0 # mV iEIF (Na inactivation slope)
lifieifsfArgs['Vi'] = -63.0 # mV iEIF (Na inactivation reversal potential)
lifieifsfArgs['R'] =  1.0/0.18 # MOhm (external current channel resistance)
lifieifsfArgs['noiseStd'] = 0.5 # mV
lifieifsfArgs['hsInit'] = iEIF_hinf(lifieifsfArgs['VInit'],lifieifsfArgs['Vi'],lifieifsfArgs['ki']) # mS iEIF (slow Na inactivation IC)
lifieifsfArgs['hfInit'] = iEIF_hinf(lifieifsfArgs['VInit'],lifieifsfArgs['Vi'],lifieifsfArgs['ki']) # mS iEIF (fast Na inactivation IC)
lifieifsfArgs['theta0'] = iEIF_thetafn(iEIF_VTfn(1.0/lifieifsfArgs['R'],lifieifsfArgs['ka'],lifieifsfArgs['Vc'],lifieifsfArgs['gNa'],lifieifsfArgs['ENa']),lifieifsfArgs['ka'],lifieifsfArgs['hsInit'],lifieifsfArgs['hfInit']) # VT
lifieifsfArgs['theta'] = lifieifsfArgs['theta0']
lifieifsfArgs['thetaInit'] = lifieifsfArgs['theta0']
lifieifsfArgs['I0'] = 7.0 # nA (external current)

lifieifsfArgs_mod1['neuronType'] = 'LIFiEIFsf' # or
lifieifsfArgs_mod1['VInit'] = -67.0 # mV (initial condition of V)
lifieifsfArgs_mod1['Vr'] = -48.0 # mV (reset potential)
lifieifsfArgs_mod1['Vb'] = -67.0 # mV (baseline potential -- bias)
lifieifsfArgs_mod1['Vc'] = -28.4391 # Va of the original iEIF mV (Na activation potential, chosen such that initial threshold = -41 mV)
lifieifsfArgs_mod1['ENa'] = 50.0 # mV iEIF (Na reversal potential)
lifieifsfArgs_mod1['gNa'] = 0.036 # mS iEIF (Na conductance)
lifieifsfArgs_mod1['tau'] =  38.0 # ms (membrane time constant)
lifieifsfArgs_mod1['tauf'] = 15.0 # ms iEIF (fast inactivation time scale)
lifieifsfArgs_mod1['taus'] = 800.0 # ms iEIF (slow inactivation time scale)
lifieifsfArgs_mod1['ka'] = 4 # mV iEIF (Na+ activation slope factor)
lifieifsfArgs_mod1['ki'] = 6.0 # mV iEIF (Na inactivation slope)
lifieifsfArgs_mod1['Vi'] = -55.0 # mV iEIF (Na inactivation reversal potential)
lifieifsfArgs_mod1['R'] =  1.0/0.0237 # MOhm (external current channel resistance) # assuming C = 0.9 muF / cm2, gL = 0.0237, or R = 42.222
lifieifsfArgs_mod1['noiseStd'] = 0.5 # mV
lifieifsfArgs_mod1['hsInit'] = iEIF_hinf(lifieifsfArgs_mod1['VInit'],lifieifsfArgs_mod1['Vi'],lifieifsfArgs_mod1['ki']) # mS iEIF (slow Na inactivation IC)
lifieifsfArgs_mod1['hfInit'] = iEIF_hinf(lifieifsfArgs_mod1['VInit'],lifieifsfArgs_mod1['Vi'],lifieifsfArgs_mod1['ki']) # mS iEIF (fast Na inactivation IC)
lifieifsfArgs_mod1['theta0'] = iEIF_thetafn(iEIF_VTfn(1.0/lifieifsfArgs_mod1['R'],lifieifsfArgs_mod1['ka'],lifieifsfArgs_mod1['Vc'],lifieifsfArgs_mod1['gNa'],lifieifsfArgs_mod1['ENa']),lifieifsfArgs_mod1['ka'],lifieifsfArgs_mod1['hsInit'],lifieifsfArgs_mod1['hfInit']) # VT
lifieifsfArgs_mod1['theta'] = lifieifsfArgs_mod1['theta0']
lifieifsfArgs_mod1['thetaInit'] = lifieifsfArgs_mod1['theta0']
lifieifsfArgs_mod1['I0'] = 1.5 # nA (external current)

lifieifArgs['neuronType'] = 'LIFiEIF' # or
lifieifArgs['VInit'] = -67.0 # mV (initial condition of V)
lifieifArgs['Vr'] = -48.0 # mV (reset potential)
lifieifArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifieifArgs['Vc'] = -36.6095 # Va of the original iEIF mV (Na activation potential, chosen such that initial threshold = -41 mV)
lifieifArgs['ENa'] = 50.0 # mV iEIF (Na reversal potential)
lifieifArgs['gNa'] = 0.036 # mS iEIF (Na conductance)
lifieifArgs['tau'] =  38.0 # ms (membrane time constant)
lifieifArgs['taus'] = 500.0 # ms iEIF (slow inactivation time scale)
lifieifArgs['ka'] = 6 # mV iEIF (Na+ activation slope factor)
lifieifArgs['ki'] = 10.0 # mV iEIF (Na inactivation slope)
lifieifArgs['Vi'] = -90.0 # mV iEIF (Na inactivation reversal potential)
lifieifArgs['R'] =  1.0/0.0237 # MOhm (external current channel resistance) # assuming C = 0.9 muF / cm2, gL = 0.0237, or R = 42.222
lifieifArgs['noiseStd'] = 0.5 # mV
lifieifArgs['hsInit'] = iEIF_hinf(lifieifArgs['VInit'],lifieifArgs['Vi'],lifieifArgs['ki']) # mS iEIF (slow Na inactivation IC)
lifieifArgs['theta0'] = iEIF_thetafn(iEIF_VTfn(1.0/lifieifArgs['R'],lifieifArgs['ka'],lifieifArgs['Vc'],lifieifArgs['gNa'],lifieifArgs['ENa']),lifieifArgs['ka'],lifieifArgs['hsInit'],1.0) # VT
lifieifArgs['theta'] = lifieifArgs['theta0']
lifieifArgs['thetaInit'] = lifieifArgs['theta0']
lifieifArgs['I0'] = 1.1 # nA (external current)
#print('theta0 = %f'%lifieifArgs['theta0'])

qifArgs['neuronType'] = 'QIF' # or
qifArgs['VInit'] = -70.0 # mV
qifArgs['thetaInit'] = -50.0 # mV
qifArgs['Vr'] = -70.0 # mV
qifArgs['Vb'] = -65.0 # mV
qifArgs['Vc'] = -55.0 # mV
qifArgs['theta'] = -50.0 # mV
qifArgs['tau'] =  10.0 # ms
qifArgs['R'] =  1.0 # MOhm
qifArgs['theta0'] =  -50.0 # mV
qifArgs['tauTheta'] =  600.0 # ms
qifArgs['DeltaTheta'] =  3.0 # mV
qifArgs['gInit'] = 0.0 # nA (initial condition on K conductance)
qifArgs['EK'] = 0.0 # mV (K resting potential)
qifArgs['tauK'] = 10.0 # ms (K current recovery time)
qifArgs['DeltaGK'] = 0.0 # nA (increase in K conductance on spike)
qifArgs['noiseStd'] = 1.0 # mV
qifArgs['I0'] = 2.6 # nA

qifdtArgs['neuronType'] = 'QIFDT' # or
qifdtArgs['VInit'] = -70.0 # mV
qifdtArgs['thetaInit'] = -50.0 # mV
qifdtArgs['Vr'] = -70.0 # mV
qifdtArgs['Vb'] = -65.0 # mV
qifdtArgs['Vc'] = -55.0 # mV
qifdtArgs['theta'] = -50.0 # mV
qifdtArgs['tau'] =  10.0 # ms
qifdtArgs['R'] =  1.0 # MOhm
qifdtArgs['theta0'] =  -50.0 # mV
qifdtArgs['tauTheta'] =  600.0 # ms
qifdtArgs['DeltaTheta'] =  3.0 # mV
qifdtArgs['gInit'] = 0.0 # nA (initial condition on K conductance)
qifdtArgs['EK'] = 0.0 # mV (K resting potential)
qifdtArgs['tauK'] = 10.0 # ms (K current recovery time)
qifdtArgs['DeltaGK'] = 0.0 # nA (increase in K conductance on spike)
qifdtArgs['noiseStd'] = 1.0 # mV
qifdtArgs['I0'] = 2.6 # nA

lifdtkArgs['neuronType'] = 'LIFDTK' # or
lifdtkArgs['VInit'] = -67.0 # mV (initial condition of V)
lifdtkArgs['Vr'] = -48.0 # mV (reset potential)
lifdtkArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdtkArgs['Vc'] = 0.0 # mV (nothing, used for QIF)
lifdtkArgs['tau'] =  38.0 # ms (membrane time constant)
lifdtkArgs['R'] =  1.0 # MOhm (external current channel resistance)
lifdtkArgs['theta0'] =  -40.0 # mV (resting state threshold)
lifdtkArgs['tauTheta'] =  600.0 # ms (threshold time constant)
lifdtkArgs['DeltaTheta'] =  2.5 # mV (threshold increment on spike)
lifdtkArgs['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdtkArgs['EK'] = -85.0 # mV (K resting potential)
lifdtkArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtkArgs['DeltaGK'] = 0.5 # nA (increase in K conductance on spike)
lifdtkArgs['noiseStd'] = 0.5 # mV
lifdtkArgs['I0'] = 40.0 # nA (external current)
lifdtkArgs['thetaInit'] = lifdtkArgs['theta0']
lifdtkArgs['theta'] =  lifdtkArgs['theta0']

lifdtboundkArgs['neuronType'] = 'LIFDTBoundK' # or
lifdtboundkArgs['VInit'] = -67.0 # mV (initial condition of V)
lifdtboundkArgs['Vr'] = -48.0 # mV (reset potential) # -45
lifdtboundkArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdtboundkArgs['tau'] =  38.0 # ms (membrane time constant)
lifdtboundkArgs['R'] =  150.0 # MOhm (external current channel resistance)
lifdtboundkArgs['theta0'] =  -41.0 # mV (resting state threshold)
lifdtboundkArgs['thetaMax'] =  -20.0 # mV (upper limit for growth of theta) # -20
lifdtboundkArgs['tauTheta'] =  540.0 # ms (threshold time constant)
lifdtboundkArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
lifdtboundkArgs['gInit'] = 0.0 / lifdtboundkArgs['R'] # nA (initial condition on K conductance)
lifdtboundkArgs['EK'] = -70.0 # mV (K resting potential)
lifdtboundkArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtboundkArgs['DeltaGK'] = 0.1 / lifdtboundkArgs['R'] # nA (increase in K conductance on spike) # 0.5
lifdtboundkArgs['noiseStd'] = 0.5 # mV
lifdtboundkArgs['I0'] = 40.0 / lifdtboundkArgs['R'] # nA (external current)
lifdtboundkArgs['thetaInit'] = lifdtboundkArgs['theta0'] # mV (initial condition of the threshold)
lifdtboundkArgs['theta'] =  lifdtboundkArgs['theta0'] # mV (initial threshold, same as thetaInit)

lifdtboundklrArgs['neuronType'] = 'LIFDTBoundKLR' # or
lifdtboundklrArgs['VInit'] = -67.0 # mV (initial condition of V)
lifdtboundklrArgs['Vr'] = -48.0 # mV (reset potential) # -45
lifdtboundklrArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdtboundklrArgs['tau'] =  38.0 # ms (membrane time constant)
lifdtboundklrArgs['R'] =  150.0 # MOhm (external current channel resistance)
lifdtboundklrArgs['theta0'] =  -41.0 # mV (resting state threshold)
lifdtboundklrArgs['thetaMax'] =  -20.0 # mV (upper limit for growth of theta) # -20
lifdtboundklrArgs['tauTheta'] =  540.0 # ms (threshold time constant)
lifdtboundklrArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
lifdtboundklrArgs['gInit'] = 0.0 / lifdtboundklrArgs['R'] # nA (initial condition on K conductance)
lifdtboundklrArgs['EK'] = -70.0 # mV (K resting potential)
lifdtboundklrArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtboundklrArgs['DeltaGK'] = 0.5 / lifdtboundklrArgs['R'] # nA (increase in K conductance on spike) # 0.5
lifdtboundklrArgs['noiseStd'] = 0.5 # mV
lifdtboundklrArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
lifdtboundklrArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
lifdtboundklrArgs['I0'] = 40.0 / lifdtboundklrArgs['R'] # nA (external current)
lifdtboundklrArgs['thetaInit'] = lifdtboundklrArgs['theta0'] # mV (initial condition of the threshold)
lifdtboundklrArgs['theta'] =  lifdtboundklrArgs['theta0'] # mV (initial threshold, same as thetaInit)

adeifhdtboundsigklrArgs = lifdtboundklrArgs.copy()
adeifhdtboundsigklrArgs['neuronType'] = 'AdEIFHDTBoundSigKLR' # or
adeifhdtboundsigklrArgs['VInit'] = -67.0 # mV (initial condition of V)
adeifhdtboundsigklrArgs['Vr'] = -48.0 # mV (reset potential) # -45
adeifhdtboundsigklrArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
adeifhdtboundsigklrArgs['VT'] = -45.0 # mV (exponential half potential)
adeifhdtboundsigklrArgs['DeltaT'] = 10.0 # mV (slope of Na activation )
adeifhdtboundsigklrArgs['tau'] =  38.0 # ms (membrane time constant)
adeifhdtboundsigklrArgs['R'] =  150.0 # MOhm (external current channel resistance)
adeifhdtboundsigklrArgs['Rth'] =  13.43 # MOhm (threshold resistance to external input, fitted from th vs. I from MC experiments) -- other value: 8.0
adeifhdtboundsigklrArgs['lf_s'] =  13.44 # mV (sigmoid slope, fitted from th vs. I from MC experiments) -- other value 20.0
adeifhdtboundsigklrArgs['lf_I0'] =  0.11 # nA (threshold baseline current from th vs. I from MC experiments) -- other value: 0.15
adeifhdtboundsigklrArgs['theta0'] =  -46.67 # mV (resting threshold, fitted for th vs. I from MC experiments)
adeifhdtboundsigklrArgs['tauRiseTheta'] = 20.0 # ms # from 1 to 20 ms
adeifhdtboundsigklrArgs['thetaMax'] =  20.0 # mV (upper limit for growth of theta) # -20
adeifhdtboundsigklrArgs['tauTheta'] =  540.0 # ms (threshold time constant)
adeifhdtboundsigklrArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
adeifhdtboundsigklrArgs['gInit'] = 0.0 / adeifhdtboundsigklrArgs['R'] # nA (initial condition on K conductance)
adeifhdtboundsigklrArgs['EK'] = -70.0 # mV (K resting potential)
adeifhdtboundsigklrArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
adeifhdtboundsigklrArgs['DeltaGK'] = 0.5 / adeifhdtboundsigklrArgs['R'] # nA (increase in K conductance on spike) # 0.5
adeifhdtboundsigklrArgs['noiseStd'] = 0.5 # mV
adeifhdtboundsigklrArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
adeifhdtboundsigklrArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
adeifhdtboundsigklrArgs['wInit'] = numpy.nan # init cond adex
adeifhdtboundsigklrArgs['gW'] = 0.001 # adapt current conductance
adeifhdtboundsigklrArgs['DeltaW'] = 0.01 # adapt current increase
adeifhdtboundsigklrArgs['tauW'] = 20.0 # adapt current time const
adeifhdtboundsigklrArgs['I0'] = 0.15 # nA (external current)
adeifhdtboundsigklrArgs['thetaInit'] = numpy.nan # mV (initial condition of the threshold) NaN means starting from resting state
adeifhdtboundsigklrArgs['thetasInit'] = 0.0 # mV (initial condition of the threshold) NaN means starting from resting state
adeifhdtboundsigklrArgs['theta'] =  adeifhdtboundsigklrArgs['theta0'] # mV (initial threshold, same as thetaInit)

eifhdtboundsigklrArgs = lifdtboundklrArgs.copy()
eifhdtboundsigklrArgs['neuronType'] = 'EIFHDTBoundSigKLR' # or
eifhdtboundsigklrArgs['VInit'] = -67.0 # mV (initial condition of V)
eifhdtboundsigklrArgs['Vr'] = -48.0 # mV (reset potential) # -45
eifhdtboundsigklrArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
eifhdtboundsigklrArgs['VT'] = -45.0 # mV (exponential half potential)
eifhdtboundsigklrArgs['DeltaT'] = 10.0 # mV (slope of Na activation )
eifhdtboundsigklrArgs['tau'] =  38.0 # ms (membrane time constant)
eifhdtboundsigklrArgs['R'] =  150.0 # MOhm (external current channel resistance)
eifhdtboundsigklrArgs['Rth'] =  13.43 # MOhm (threshold resistance to external input, fitted from th vs. I from MC experiments) -- other value: 8.0
eifhdtboundsigklrArgs['lf_s'] =  13.44 # mV (sigmoid slope, fitted from th vs. I from MC experiments) -- other value: 20.0
eifhdtboundsigklrArgs['lf_I0'] =  0.11 # nA (threshold baseline current from th vs. I from MC experiments) -- other value: 0.15
eifhdtboundsigklrArgs['theta0'] =  -46.67 # mV (resting threshold, fitted for th vs. I from MC experiments)
eifhdtboundsigklrArgs['tauRiseTheta'] = 20.0 # ms # from 1 to 20 ms
eifhdtboundsigklrArgs['thetaMax'] =  20.0 # mV (upper limit for growth of theta) # -20
eifhdtboundsigklrArgs['tauTheta'] =  540.0 # ms (threshold time constant)
eifhdtboundsigklrArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
eifhdtboundsigklrArgs['gInit'] = 0.0 / eifhdtboundsigklrArgs['R'] # nA (initial condition on K conductance)
eifhdtboundsigklrArgs['EK'] = -70.0 # mV (K resting potential)
eifhdtboundsigklrArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
eifhdtboundsigklrArgs['DeltaGK'] = 0.5 / eifhdtboundsigklrArgs['R'] # nA (increase in K conductance on spike) # 0.5
eifhdtboundsigklrArgs['noiseStd'] = 0.5 # mV
eifhdtboundsigklrArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
eifhdtboundsigklrArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
eifhdtboundsigklrArgs['I0'] = 0.15 # nA (external current)
eifhdtboundsigklrArgs['thetaInit'] = numpy.nan # mV (initial condition of the threshold) NaN means starting from resting state
eifhdtboundsigklrArgs['thetasInit'] = 0.0 # mV (initial condition of the threshold) NaN means starting from resting state
eifhdtboundsigklrArgs['theta'] =  eifhdtboundsigklrArgs['theta0'] # mV (initial threshold, same as thetaInit)

eifhdtboundsigklriaArgs = lifdtboundklrArgs.copy()
eifhdtboundsigklriaArgs['neuronType'] = 'EIFHDTBoundSigKLRIA' # or
eifhdtboundsigklriaArgs['VInit'] = -67.0 # mV (initial condition of V)
eifhdtboundsigklriaArgs['Vr'] = -48.0 # mV (reset potential) # -45
eifhdtboundsigklriaArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
eifhdtboundsigklriaArgs['VT'] = -45.0 # mV (exponential half potential)
eifhdtboundsigklriaArgs['DeltaT'] = 10.0 # mV (slope of Na activation )
eifhdtboundsigklriaArgs['tau'] =  38.0 # ms (membrane time constant)
eifhdtboundsigklriaArgs['R'] =  150.0 # MOhm (external current channel resistance)
eifhdtboundsigklriaArgs['Rth'] =  13.43 # MOhm (threshold resistance to external input, fitted from th vs. I from MC experiments) -- other value: 8.0
eifhdtboundsigklriaArgs['lf_s'] =  13.44 # mV (sigmoid slope, fitted from th vs. I from MC experiments) -- other value 20.0
eifhdtboundsigklriaArgs['lf_I0'] =  0.11 # nA (threshold baseline current from th vs. I from MC experiments) -- other value: 0.15
eifhdtboundsigklriaArgs['theta0'] =  -46.67 # mV (resting threshold, fitted for th vs. I from MC experiments)
eifhdtboundsigklriaArgs['tauRiseTheta'] = 20.0 # ms # from 1 to 20 ms
eifhdtboundsigklriaArgs['thetaMax'] =  20.0 # mV (upper limit for growth of theta) # -20
eifhdtboundsigklriaArgs['tauTheta'] =  540.0 # ms (threshold time constant)
eifhdtboundsigklriaArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
eifhdtboundsigklriaArgs['gInit'] = 0.0 / eifhdtboundsigklriaArgs['R'] # nA (initial condition on K conductance)
eifhdtboundsigklriaArgs['EK'] = -70.0 # mV (K resting potential)
eifhdtboundsigklriaArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
eifhdtboundsigklriaArgs['DeltaGK'] = 0.5 / eifhdtboundsigklriaArgs['R'] # nA (increase in K conductance on spike) # 0.5
eifhdtboundsigklriaArgs['noiseStd'] = 0.5 # mV
eifhdtboundsigklriaArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
eifhdtboundsigklriaArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
eifhdtboundsigklriaArgs['I0'] = 0.15 # nA (external current)
eifhdtboundsigklriaArgs['thetaInit'] = numpy.nan # mV (initial condition of the threshold) NaN means starting from resting state
eifhdtboundsigklriaArgs['thetasInit'] = 0.0 # mV (initial condition of the threshold) NaN means starting from resting state
eifhdtboundsigklriaArgs['theta'] =  eifhdtboundsigklriaArgs['theta0'] # mV (initial threshold, same as thetaInit)
eifhdtboundsigklriaArgs['gIA'] = 0.02 # max conductance of the IA current
eifhdtboundsigklriaArgs['hIAInit'] = 0.0 # IA inactivation current initial condition (from Harkin paper draft)
eifhdtboundsigklriaArgs['AmIA'] = 1.61 # activation amplitude IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifhdtboundsigklriaArgs['kmIA'] = 0.0985 # 1/mV, activation slope IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifhdtboundsigklriaArgs['VmIA'] = -23.7 # mV, activation half potential IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifhdtboundsigklriaArgs['AhIA'] = 1.03 # inactivation amplitude IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifhdtboundsigklriaArgs['khIA'] = -0.165# inactivation slope IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifhdtboundsigklriaArgs['VhIA'] = -59.2# inactivation half potential IA current -- fitted for serotonergic neurons (from Harkin paper draft)
eifhdtboundsigklriaArgs['tauhIA'] = 43.0#ms inactivation timescale IA current -- fitted for serotonergic neurons (from Harkin paper draft; activation was about 7 ms)
eifhdtboundsigklriaArgs['EKIA'] = -85.0# mV reversal K potential for IA current -- fitted for serotonergic neurons (from Harkin paper draft)


lifdltboundklrArgs['neuronType'] = 'LIFDLTBoundKLR' # or
lifdltboundklrArgs['Vr'] = -48.0 # mV (reset potential) # -45
lifdltboundklrArgs['Vb'] = -70.0 # mV (baseline potential -- bias)
lifdltboundklrArgs['tau'] =  38.0 # ms (membrane time constant)
lifdltboundklrArgs['R'] =  150.0 # MOhm (external current channel resistance) # avg value is 150, it goes from 100 to 200 MOhm in experiments
lifdltboundklrArgs['Vc'] = -20.0 # mV (potential in which the theta starts to grow linearly with V)
lifdltboundklrArgs['Vo'] = -55.0 # mV (potential in which the theta starts to grow linearly with V)
lifdltboundklrArgs['theta0'] =  -57.2 # mV (resting state threshold) -- fitted from MATLAB
lifdltboundklrArgs['Rth'] =  100.0 # MOhm (proportionality between theta0 and Iext) -- fitted from MATLAB
lifdltboundklrArgs['thetaMax'] =  -20.0 # mV (upper limit for growth of theta) # -20
lifdltboundklrArgs['tauTheta'] =  540.0 # ms (threshold time constant)
lifdltboundklrArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
lifdltboundklrArgs['gInit'] = 0.0 / lifdltboundklrArgs['R'] # nA (initial condition on K conductance)
lifdltboundklrArgs['EK'] = -90.0 # mV (K resting potential)
lifdltboundklrArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdltboundklrArgs['DeltaGK'] = 0.5 / lifdltboundklrArgs['R'] # nA (increase in K conductance on spike) # 0.5
lifdltboundklrArgs['noiseStd'] = 0.5 # mV
lifdltboundklrArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
lifdltboundklrArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
lifdltboundklrArgs['I0'] = 30.0 / lifdltboundklrArgs['R'] # nA/MOhm (external current)
lifdltboundklrArgs['VInit'] = lifdltboundklrArgs['Vb'] # mV (initial condition of V)
lifdltboundklrArgs['thetaInit'] = lifdltboundklrArgs['theta0'] # mV (initial condition of the threshold)
lifdltboundklrArgs['theta'] =  lifdltboundklrArgs['theta0'] # mV (initial threshold, same as thetaInit)

lifdltboundklriaArgs['neuronType'] = 'LIFDLTBoundKLRIA' # or
lifdltboundklriaArgs['Vr'] = -48.0 # mV (reset potential) # -45
lifdltboundklriaArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdltboundklriaArgs['tau'] =  38.0 # ms (membrane time constant)
lifdltboundklriaArgs['R'] =  50.0 # MOhm (external current channel resistance) # avg value is 150, it goes from 100 to 200 MOhm in experiments
lifdltboundklriaArgs['theta0'] = -57.2 # mV (resting state threshold) -- fitted from MATLAB
lifdltboundklriaArgs['Rth'] =  100.0 # MOhm (proportionality between theta0 and Iext) -- fitted from MATLAB
lifdltboundklriaArgs['thetaMax'] =  -20.0 # mV (upper limit for growth of theta) # -20
lifdltboundklriaArgs['tauTheta'] =  540.0 # ms (threshold time constant)
lifdltboundklriaArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
lifdltboundklriaArgs['gInit'] = 0.0 / lifdltboundklriaArgs['R'] # nA (initial condition on K conductance)
lifdltboundklriaArgs['EK'] = -90.0 # mV (K resting potential)
lifdltboundklriaArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdltboundklriaArgs['DeltaGK'] = 0.5 / lifdltboundklriaArgs['R'] # nA (increase in K conductance on spike) # 0.5
lifdltboundklriaArgs['noiseStd'] = 0.5 # mV
lifdltboundklriaArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
lifdltboundklriaArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
lifdltboundklriaArgs['mIAInit'] = numpy.nan#0.0 # the IA current starts closed
lifdltboundklriaArgs['gIA'] = 0.5 # max conductance of the IA current
lifdltboundklriaArgs['tauIA'] = 7.0 # ms time constant of the IA current
lifdltboundklriaArgs['VIA'] = -26.0 # opening potential of the IA current
lifdltboundklriaArgs['kIA'] = 1.0 # slope of the IA opening sigmoid
lifdltboundklriaArgs['I0'] = 0.58 # nA/MOhm (external current)
lifdltboundklriaArgs['VInit'] = numpy.nan#lifdltboundklriaArgs['Vb'] # mV (initial condition of V)
lifdltboundklriaArgs['thetaInit'] = lifdltboundklriaArgs['theta0'] # mV (initial condition of the threshold)
lifdltboundklriaArgs['theta'] =  lifdltboundklriaArgs['theta0'] # mV (initial threshold, same as thetaInit)

lifdltArgs['neuronType'] = 'LIFDLT' # or
lifdltArgs['Vr'] = -48.0 # mV (reset potential) # -45
lifdltArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdltArgs['tau'] =  38.0 # ms (membrane time constant)
lifdltArgs['R'] =  50.0 # MOhm (external current channel resistance) # avg value is 150, it goes from 100 to 200 MOhm in experiments
lifdltArgs['theta0'] = -57.2 # mV (resting state threshold) -- fitted from MATLAB
lifdltArgs['Rth'] =  100.0 # MOhm (proportionality between theta0 and Iext) -- fitted from MATLAB
lifdltArgs['tauTheta'] =  540.0 # ms (threshold time constant)
lifdltArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
lifdltArgs['noiseStd'] = 0.5 # mV
lifdltArgs['I0'] = 0.58 # nA/MOhm (external current)
lifdltArgs['VInit'] = lifdltArgs['Vb']#lifdltboundklriaArgs['Vb'] # mV (initial condition of V)
lifdltArgs['thetaInit'] = lifdltArgs['theta0'] # mV (initial condition of the threshold)
lifdltArgs['theta'] =  lifdltArgs['theta0'] # mV (initial threshold, same as thetaInit)

lifdtvboundklrArgs['neuronType'] = 'LIFDTVBoundKLR' # or
lifdtvboundklrArgs['Vr'] = -48.0 # mV (reset potential) # -45
lifdtvboundklrArgs['Vb'] = -70.0 # mV (baseline potential -- bias)
lifdtvboundklrArgs['tau'] =  38.0 # ms (membrane time constant)
lifdtvboundklrArgs['R'] =  150.0 # MOhm (external current channel resistance) # avg value is 150, it goes from 100 to 200 MOhm in experiments
#lifdtvboundklrArgs['Vc'] = -20.0 # mV (potential in which the theta starts to grow linearly with V)
#lifdtvboundklrArgs['Vo'] = -55.0 # mV (potential in which the theta starts to grow linearly with V)
lifdtvboundklrArgs['theta0'] =  -57.2 #-41.0 mV (resting state threshold) -- fitted from MATLAB
lifdtvboundklrArgs['thetaMax'] =  -20.0 # mV (upper limit for growth of theta) # -20
lifdtvboundklrArgs['tauTheta'] =  540.0 # ms (threshold time constant)
lifdtvboundklrArgs['Rth'] =  100.0 # MOhm (proportionality between theta0 and Iext) -- fitted from MATLAB # 100
lifdtvboundklrArgs['tauThetaV'] = 10.0 #lifdtvboundklrArgs['tauTheta'] # ms (theta vs. V decay time constant)
lifdtvboundklrArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
lifdtvboundklrArgs['gInit'] = 0.0 / lifdtvboundklrArgs['R'] # nA (initial condition on K conductance)
lifdtvboundklrArgs['EK'] = -70.0 # mV (K resting potential)
lifdtvboundklrArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtvboundklrArgs['DeltaGK'] = 0.1 / lifdtvboundklrArgs['R'] # nA (increase in K conductance on spike) # 0.5
lifdtvboundklrArgs['noiseStd'] = 0.5 # mV
lifdtvboundklrArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike -- fitted value was 0.8); 0.8
lifdtvboundklrArgs['deltaV'] = 2.0 # mV (decrement in V following a spike -- fitted value was 12 mV); 2 mV
lifdtvboundklrArgs['I0'] = 40.0 / lifdtvboundklrArgs['R'] # nA/MOhm (external current)
lifdtvboundklrArgs['VInit'] = lifdtvboundklrArgs['Vb'] # mV (initial condition of V)
lifdtvboundklrArgs['thetaInit'] = lifdtvboundklrArgs['theta0'] # mV (initial condition of the threshold)
lifdtvboundklrArgs['theta'] =  lifdtvboundklrArgs['theta0'] # mV (initial threshold, same as thetaInit)

#lifdtvboundklrArgs['neuronType'] = 'LIFDTVBoundKLR' # or
#lifdtvboundklrArgs['VInit'] = -67.0 # mV (initial condition of V)
#lifdtvboundklrArgs['Vr'] = -48.0 # mV (reset potential) # -45
#lifdtvboundklrArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
#lifdtvboundklrArgs['tau'] =  38.0 # ms (membrane time constant)
#lifdtvboundklrArgs['R'] =  1.0 # MOhm (external current channel resistance)
#lifdtvboundklrArgs['theta0'] =  -41.0 # mV (resting state threshold)
#lifdtvboundklrArgs['thetaMax'] =  -20.0 # mV (upper limit for growth of theta) # -20
#lifdtvboundklrArgs['tauTheta'] =  540.0 # ms (threshold time constant)
#lifdtvboundklrArgs['DeltaTheta'] =  3.0 # mV (threshold increment on spike) # 3.0
#lifdtvboundklrArgs['gInit'] = 0.0 # nA (initial condition on K conductance)
#lifdtvboundklrArgs['EK'] = -70.0 # mV (K resting potential)
#lifdtvboundklrArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
#lifdtvboundklrArgs['DeltaGK'] = 0.5 # nA (increase in K conductance on spike) # 0.5
#lifdtvboundklrArgs['noiseStd'] = 0.5 # mV
#lifdtvboundklrArgs['rv'] = 0.8 # (slope of V following a spike vs. V before spike)
#lifdtvboundklrArgs['deltaV'] = 12.0 # mV (decrement in V following a spike)
#lifdtvboundklrArgs['aThetaV'] = 1.0/600.0 # 1/ms (proportionality constant of theta vs. V)
#lifdtvboundklrArgs['tauThetaV'] = 600.0 # ms (theta vs. V decay time constant)
#lifdtvboundklrArgs['I0'] = 50.0 # nA (external current)
#lifdtvboundklrArgs['thetaInit'] = 0.0 # mV (initial condition of the threshold)
#lifdtvboundklrArgs['theta'] =  lifdtvboundklrArgs['thetaInit'] # mV (initial threshold, same as thetaInit)

lifdtkboundArgs['neuronType'] = 'LIFDTKBound' # or
lifdtkboundArgs['VInit'] = -67.0 # mV (initial condition of V)
lifdtkboundArgs['Vr'] = -48.0 # mV (reset potential) # -45
lifdtkboundArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdtkboundArgs['tau'] =  38.0 # ms (membrane time constant)
lifdtkboundArgs['R'] =  1.0 # MOhm (external current channel resistance)
lifdtkboundArgs['theta0'] =  -41.0 # mV (resting state threshold)
lifdtkboundArgs['gMax'] =  0.6 # mV (upper limit for growth of theta) # -20
lifdtkboundArgs['tauTheta'] =  540.0 # ms (threshold time constant)
lifdtkboundArgs['DeltaTheta'] =  2.5 # mV (threshold increment on spike) # 3.0
lifdtkboundArgs['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdtkboundArgs['EK'] = -70.0 # mV (K resting potential)
lifdtkboundArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtkboundArgs['DeltaGK'] = 0.5 # nA (increase in K conductance on spike) # 0.8
lifdtkboundArgs['noiseStd'] = 0.5 # mV
lifdtkboundArgs['I0'] = 35.0 # nA (external current)
lifdtkboundArgs['thetaInit'] = lifdtkboundArgs['theta0'] # mV (initial condition of the threshold)
lifdtkboundArgs['theta'] = lifdtkboundArgs['theta0'] # mV (initial threshold, same as thetaInit)

lifdt2kArgs['neuronType'] = 'LIFDT2K' # or
lifdt2kArgs['VInit'] = -67.0 # mV (initial condition of V)
lifdt2kArgs['thetaInit'] = -40.0 # mV (initial condition of the threshold)
lifdt2kArgs['Vr'] = -42.0 # mV (reset potential)
lifdt2kArgs['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdt2kArgs['Vc'] = 0.0 # mV (nothing, used for QIF)
lifdt2kArgs['theta'] =  -40.0 # mV (initial threshold, same as thetaInit)
lifdt2kArgs['tau'] =  38.0 # ms (membrane time constant)
lifdt2kArgs['R'] =  1.0 # MOhm (external current channel resistance)
lifdt2kArgs['theta0'] =  -40.0 # mV (resting state threshold)
lifdt2kArgs['tauTheta'] =  600.0 # ms (threshold time constant)
lifdt2kArgs['DeltaTheta'] =  2.5 # mV (threshold increment on spike)
lifdt2kArgs['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdt2kArgs['EK'] = -85.0 # mV (K resting potential)
lifdt2kArgs['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdt2kArgs['DeltaGK'] = 1.0 # nA (increase in K conductance on spike)
lifdt2kArgs['g2Init'] = 0.0 # nA (initial condition on K conductance)
lifdt2kArgs['E2'] = -55.0 # mV (K resting potential)
lifdt2kArgs['tau2'] = 10.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdt2kArgs['DeltaG2'] = 2.0 # nA (increase in K conductance on spike)
lifdt2kArgs['noiseStd'] = 0.5 # mV
lifdt2kArgs['I0'] = 40.0 # nA (external current)

lifdtkzArgs_mod1['neuronType'] = 'LIFDTKz' # or
lifdtkzArgs_mod1['VInit'] = -67.0 # mV (initial condition of V)
lifdtkzArgs_mod1['thetaInit'] = -40.0 # mV (initial condition of the threshold)
lifdtkzArgs_mod1['Vr'] = -48.0 # mV (reset potential)
lifdtkzArgs_mod1['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdtkzArgs_mod1['Vc'] = 0.0 # mV (nothing, used for QIF)
lifdtkzArgs_mod1['theta'] =  -40.0 # mV (initial threshold, same as thetaInit)
lifdtkzArgs_mod1['tau'] =  38.0 # ms (membrane time constant)
lifdtkzArgs_mod1['R'] =  1.0 # MOhm (external current channel resistance)
lifdtkzArgs_mod1['theta0'] =  -40.0 # mV (resting state threshold)
lifdtkzArgs_mod1['tauTheta'] =  600.0 # ms (threshold time constant)
lifdtkzArgs_mod1['DeltaTheta'] =  2.5 # mV (threshold increment on spike)
lifdtkzArgs_mod1['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdtkzArgs_mod1['EK'] = -85.0 # mV (K resting potential)
lifdtkzArgs_mod1['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtkzArgs_mod1['DeltaGK'] = 0.5 # nA (increase in K conductance on spike)
lifdtkzArgs_mod1['E2'] = -80.0 # mV (K resting potential)
lifdtkzArgs_mod1['tau2'] = 100.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtkzArgs_mod1['G2'] = 1.0 # nA (increase in K conductance on spike)
lifdtkzArgs_mod1['g2Init'] = lifdtkzArgs_mod1['G2'] * (lifdtkzArgs_mod1['E2'] - lifdtkzArgs_mod1['Vb'])
lifdtkzArgs_mod1['noiseStd'] = 0.5 # mV
lifdtkzArgs_mod1['I0'] = 70.0 # nA (external current)

lifdtkzArgs_mod2['neuronType'] = 'LIFDTKz' # or
lifdtkzArgs_mod2['VInit'] = -67.0 # mV (initial condition of V)
lifdtkzArgs_mod2['thetaInit'] = -40.0 # mV (initial condition of the threshold)
lifdtkzArgs_mod2['Vr'] = -48.0 # mV (reset potential)
lifdtkzArgs_mod2['Vb'] = -67.0 # mV (baseline potential -- bias)
lifdtkzArgs_mod2['Vc'] = 0.0 # mV (nothing, used for QIF)
lifdtkzArgs_mod2['theta'] =  -40.0 # mV (initial threshold, same as thetaInit)
lifdtkzArgs_mod2['tau'] =  38.0 # ms (membrane time constant)
lifdtkzArgs_mod2['R'] =  1.0 # MOhm (external current channel resistance)
lifdtkzArgs_mod2['theta0'] =  -40.0 # mV (resting state threshold)
lifdtkzArgs_mod2['tauTheta'] =  600.0 # ms (threshold time constant)
lifdtkzArgs_mod2['DeltaTheta'] =  2.5 # mV (threshold increment on spike)
lifdtkzArgs_mod2['gInit'] = 0.0 # nA (initial condition on K conductance)
lifdtkzArgs_mod2['EK'] = -85.0 # mV (K resting potential)
lifdtkzArgs_mod2['tauK'] = 40.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtkzArgs_mod2['DeltaGK'] = 0.5 # nA (increase in K conductance on spike)
lifdtkzArgs_mod2['E2'] = -80.0 # mV (K resting potential)
lifdtkzArgs_mod2['tau2'] = 100.0 # ms (K current recovery time) -- 20 ms makes the theta-theta0 curve saturate later and makes it more steep
lifdtkzArgs_mod2['G2'] = 1.0 # nA (increase in K conductance on spike)
lifdtkzArgs_mod2['g2Init'] = lifdtkzArgs_mod2['G2'] * (lifdtkzArgs_mod2['E2'] - lifdtkzArgs_mod2['Vb'])
lifdtkzArgs_mod2['noiseStd'] = 0.5 # mV
lifdtkzArgs_mod2['I0'] = 70.0 # nA (external current)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Neuron factory
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def GetNeuron(neuronArgs):
    # create neuron
    if neuronArgs['neuronType'] == 'LIF':
        return LIF(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIF.__init__)))
    elif neuronArgs['neuronType'] == 'LIFex':
        return LIFex(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFex.__init__)))
    elif neuronArgs['neuronType'] == 'LIFAdEx':
        return LIFAdEx(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFAdEx.__init__)))
    elif neuronArgs['neuronType'] == 'LIFiEIFsf':
        return LIFiEIFsf(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFiEIFsf.__init__)))
    elif neuronArgs['neuronType'] == 'LIFiEIF':
        return LIFiEIF(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFiEIF.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDT':
        return LIFDT(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDT.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDLT':
        return LIFDLT(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDLT.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDTK':
        return LIFDTK(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDTK.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDTKBound':
        return LIFDTKBounded(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDTKBounded.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDTBoundK':
        return LIFDTBoundedK(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDTBoundedK.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDTBoundKLR':
        return LIFDTBoundedKLR(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDTBoundedKLR.__init__)))
    elif neuronArgs['neuronType'] == 'EIFDT':
        return EIFDT(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(EIFDT.__init__)))
    elif neuronArgs['neuronType'] == 'EIFDTBoundKLR':
        return EIFDTBoundKLR(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(EIFDTBoundKLR.__init__)))
    elif neuronArgs['neuronType'] == 'EIFDTBoundSigKLR':
        return EIFDTBoundSigKLR(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(EIFDTBoundSigKLR.__init__)))
    elif neuronArgs['neuronType'] == 'EIFDTBoundSigVKLR':
        return EIFDTBoundSigVKLR(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(EIFDTBoundSigVKLR.__init__)))
    elif neuronArgs['neuronType'] == 'EIFDTBoundSigKLRIA':
        return EIFDTBoundSigKLRIA(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(EIFDTBoundSigKLRIA.__init__)))
    elif neuronArgs['neuronType'] == 'EIFSubDT':
        return EIFSubDT(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(EIFSubDT.__init__)))
    elif neuronArgs['neuronType'] == 'AdEIFHDTBoundSigKLR':
        return AdEIFHDTBoundSigKLR(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(AdEIFHDTBoundSigKLR.__init__)))
    elif neuronArgs['neuronType'] == 'EIFHDTBoundSigKLR':
        return EIFHDTBoundSigKLR(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(EIFHDTBoundSigKLR.__init__)))
    elif neuronArgs['neuronType'] == 'EIFHDTBoundSigKLRIA':
        return EIFHDTBoundSigKLRIA(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(EIFHDTBoundSigKLRIA.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDLTBoundKLR':
        return LIFDLTBoundedKLR(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDLTBoundedKLR.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDLTBoundKLRIA':
        return LIFDLTBoundedKLRIA(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDLTBoundedKLRIA.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDTVBoundKLR':
        return LIFDTVBoundedKLR(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDTVBoundedKLR.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDT2K':
        return LIFDT2K(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDT2K.__init__)))
    elif neuronArgs['neuronType'] == 'LIFDTKz':
        return LIFDTKz(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(LIFDTKz.__init__)))
    elif neuronArgs['neuronType'] == 'QIF':
        return QIF(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(QIF.__init__)))
    elif neuronArgs['neuronType'] == 'QIFDT':
        return QIFDT(**keep_keys(neuronArgs,['dt', 'noiseSignal'] + get_func_param(QIFDT.__init__)))
    else:
        raise ValueError("Unknown neuron")

def modify_args(args,new_values):
    if type(new_values) is type(None):
        return args
    if not new_values:
        return args
    if len(new_values) == 0:
        return args
    args = copy.deepcopy(args)
    if new_values:
        print(' ... modifying arguments:')
    for k,v in new_values.items():
        print('      %s = %s'%(k,str(v)))
        args[k] = v
    return args

def GetDefaultParamSet(paramset,neuronArgs=None,modified_paramsetArgs=None):
    if paramset == 'none':
        if neuronArgs is None:
            raise ValueError('you must determine either the paramset or the neuronArgs')
        if not ('I0' in neuronArgs.keys()):
            neuronArgs['I0'] = 40.0
        return neuronArgs
    if paramset == 'LIFDT':
        return modify_args(lifdtArgs_default,modified_paramsetArgs)
    elif paramset == 'LIFDLT':
        return modify_args(lifdltArgs,modified_paramsetArgs)
    elif paramset == 'LIFex':
        return modify_args(lifexArgs,modified_paramsetArgs)
    elif paramset == 'LIFAdEx':
        return modify_args(lifadexArgs,modified_paramsetArgs)
    elif paramset == 'LIFiEIF':
        return modify_args(lifieifArgs,modified_paramsetArgs)
    elif paramset == 'LIFiEIFsf':
        return modify_args(lifieifsfArgs,modified_paramsetArgs)
    elif paramset == 'LIFiEIFsfmod1':
        return modify_args(lifieifsfArgs_mod1,modified_paramsetArgs)
    elif paramset == 'LIFDTmod1':
        return modify_args(lifdtArgs_mod1,modified_paramsetArgs)
    elif paramset == 'LIFDTmod2':
        return modify_args(lifdtArgs_mod2,modified_paramsetArgs)
    elif paramset == 'LIFDTmod3':
        return modify_args(lifdtArgs_mod3,modified_paramsetArgs)
    elif paramset == 'LIFDTK':
        return modify_args(lifdtkArgs,modified_paramsetArgs)
    elif paramset == 'LIFDTBoundK':
        return modify_args(lifdtboundkArgs,modified_paramsetArgs)
    elif paramset == 'LIFDTBoundKLR':
        return modify_args(lifdtboundklrArgs,modified_paramsetArgs)
    elif paramset == 'AdEIFHDTBoundSigKLR':
        return modify_args(adeifhdtboundsigklrArgs,modified_paramsetArgs)
    elif paramset == 'EIFDT':
        return modify_args(eifdtArgs,modified_paramsetArgs)
    elif paramset == 'EIFDTBoundKLR':
        return modify_args(eifdtboundklrArgs,modified_paramsetArgs)
    elif paramset == 'EIFDTBoundSigKLR':
        return modify_args(eifdtboundsigklrArgs,modified_paramsetArgs)
    elif paramset == 'EIFDTBoundSigVKLR':
        return modify_args(eifdtboundsigvklrArgs,modified_paramsetArgs)
    elif paramset == 'EIFDTBoundSigKLRIA':
        return modify_args(eifdtboundsigklriaArgs,modified_paramsetArgs)
    elif paramset == 'EIFSubDT':
        return modify_args(eifsubdtArgs,modified_paramsetArgs)
    elif paramset == 'EIFHDTBoundSigKLR':
        return modify_args(eifhdtboundsigklrArgs,modified_paramsetArgs)
    elif paramset == 'EIFHDTBoundSigKLRIA':
        return modify_args(eifhdtboundsigklriaArgs,modified_paramsetArgs)
    elif paramset == 'LIFDLTBoundKLR':
        return modify_args(lifdltboundklrArgs,modified_paramsetArgs)
    elif paramset == 'LIFDLTBoundKLRIA':
        return modify_args(lifdltboundklriaArgs,modified_paramsetArgs)
    elif paramset == 'LIFDTVBoundKLR':
        return modify_args(lifdtvboundklrArgs,modified_paramsetArgs)
    elif paramset == 'LIFDTKBound':
        return modify_args(lifdtkboundArgs,modified_paramsetArgs)
    elif paramset == 'LIFDT2K':
        return modify_args(lifdt2kArgs,modified_paramsetArgs)
    elif paramset == 'LIFDTKzmod1':
        return modify_args(lifdtkzArgs_mod1,modified_paramsetArgs)
    elif paramset == 'LIFDTKzmod2':
        return modify_args(lifdtkzArgs_mod2,modified_paramsetArgs)
    elif paramset == 'QIF':
        return modify_args(qifArgs,modified_paramsetArgs)
    elif paramset == 'QIFDT':
        return modify_args(qifdtArgs,modified_paramsetArgs)
    return neuronArgs

def GetDefaultCurrentRange(paramset):
    if paramset == 'LIFDT':
        return numpy.linspace(30.0,120.0,10)
    elif paramset == 'AdEIFHDTBoundSigKLR':
        return numpy.linspace(0.06,0.4,10)
    elif paramset == 'EIFSubDT':
        return numpy.linspace(0.1,0.4,10)
    elif paramset == 'EIFDT':
        return numpy.linspace(0.1,0.4,10)
    elif paramset == 'EIFDTBoundKLR':
        return numpy.linspace(0.1,0.4,10)
    elif paramset == 'EIFDTBoundSigKLR':
        return numpy.linspace(0.06,0.3,10)
    elif paramset == 'EIFDTBoundSigVKLR':
        return numpy.linspace(0.06,0.3,10)
    elif paramset == 'EIFDTBoundSigKLRIA':
        return numpy.linspace(0.06,0.3,10)
    elif paramset == 'LIFex':
        return numpy.linspace(0.16,0.23,10)
    elif paramset == 'EIFHDTBoundSigKLR':
        return numpy.linspace(0.06,0.4,10)
    elif paramset == 'EIFHDTBoundSigKLRIA':
        return numpy.linspace(0.07,0.4,10)
    elif paramset == 'LIFDLT':
        return numpy.linspace(30.0,120.0,10)
    elif paramset == 'LIFiEIF':
        return numpy.linspace(0.7,1.4,10)
    elif paramset == 'LIFiEIFsf':
        return numpy.linspace(1.2,2.2,10)
    elif paramset == 'LIFiEIFsfmod1':
        return numpy.linspace(1.2,2.2,10)
    elif paramset == 'LIFDTmod1':
        return numpy.linspace(30.0,120.0,10)
    elif paramset == 'LIFDTmod2':
        return numpy.linspace(30.0,120.0,10)
    elif paramset == 'LIFDTmod3':
        return numpy.linspace(30.0,120.0,10)
    elif paramset == 'LIFDTK':
        return numpy.linspace(30.0,120.0,10)
    elif paramset == 'LIFDTBoundK':
        return numpy.linspace(0.2,0.8,10)
    elif paramset == 'LIFDTBoundKLR':
        return numpy.linspace(0.18,0.5,10)
    elif paramset == 'LIFDTVBoundKLR':
        return numpy.linspace(0.2,0.8,10)
    elif paramset == 'LIFDLTBoundKLR':
        return numpy.linspace(0.1,0.4,10)
    elif paramset == 'LIFDLTBoundKLRIA':
        return numpy.linspace(0.58,0.68,10)
    return 'nan'

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Noise factory
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""
def GetNoise(neuronArgs):
    # create noise
    if 'noiseType' in neuronArgs.keys():
        if neuronArgs['noiseType'] == 'white':
            return SynapticWhiteNoise(neuronArgs['noiseStd'],neuronArgs['dt'])
        elif neuronArgs['noiseType'] == 'poisson':
            return PoissonProcess(neuronArgs['noise_rate'],neuronArgs['noise_J'])
        elif neuronArgs['noiseType'] == 'synpoisson':
            return SynapticPoissonNoise(neuronArgs['noise_rate'],neuronArgs['noise_tau1'],neuronArgs['noise_tau2'],neuronArgs['noise_J'],1.0,0.0,neuronArgs['dt'])
        else:
            print('unknown noise type... returning empty 0 noise')
            return None
    if neuronArgs['noiseStd'] != 0.0:
        return SynapticWhiteNoise(neuronArgs['noiseStd'],neuronArgs['dt'])
    else:
        print('unknown noise type... returning empty 0 noise')
        return None

def GetSynapticNoise(simArgs):
    if 'synnoise_type' in simArgs:
        if simArgs['synnoise_type'] == 'synpoisson':
            return SynapticPoissonNoise(simArgs['noise_rate'],simArgs['noise_tau1'],simArgs['noise_tau2'],simArgs['noise_J'],1.0,0.0,simArgs['dt'])
        elif simArgs['synnoise_type'] == 'poisson':
            return PoissonProcess(simArgs['noise_rate'],simArgs['noise_J'])
        elif simArgs['synnoise_type'] == 'none':
            return None
    return None

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Single Neuron with receptive field
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def RunSimulation(**simArgs):
    dt = simArgs['dt']

    # create noise
    noise = GetNoise(simArgs)
    if noise is None:
        simArgs['noiseSignal'] = lambda:0.0
    else:
        simArgs['noiseSignal'] = noise.GetSignal

    # create neuron
    neuron = GetNeuron(simArgs)
    synnoise = GetSynapticNoise(simArgs)
    if synnoise:
        neuron.AddInput(synnoise)
    
    trajArgs = replace_keys(keep_keys(simArgs,['mouse_x0','mouse_y0','mouse_vx','mouse_vy','dt']),
                                                 ['mouse_x0','mouse_y0','mouse_vx','mouse_vy'],
                                                 ['x0','y0','v0x','v0y'])
    mouse = StraightTrajectory(**trajArgs)

    recfArgs = replace_keys(keep_keys(simArgs,['I0','recf_x0','recf_y0','recf_R']),
                                                 ['recf_x0','recf_y0','recf_R'],
                                                 ['x0','y0','radius'])
    recf = ReceptiveField(**recfArgs)

    recf.SetTrajectory(mouse)
    neuron.AddInput(recf)

    # setup simulation
    T = ( 2.0*numpy.abs(simArgs['mouse_x0']) + 2.0*simArgs['recf_R'] ) / numpy.linalg.norm([ simArgs['mouse_vx'], simArgs['mouse_vy'] ])
    Tsteps = int(numpy.ceil(T / dt))

    # creating data record variables
    ntrials = simArgs['ntrials']
    ts = numpy.arange(Tsteps)
    V = [numpy.zeros(Tsteps) for k in range(ntrials)]
    th = [numpy.zeros(Tsteps) for k in range(ntrials)]
    g1 = [numpy.zeros(Tsteps) for k in range(ntrials)]
    g2 = [numpy.zeros(Tsteps) for k in range(ntrials)]
    x = numpy.zeros(Tsteps)
    y = numpy.zeros(Tsteps)
    I = numpy.zeros(Tsteps)

    # run :)
    for k in range(ntrials):
        mouse.Reset()
        neuron.Reset()
        recf.Reset()

        for t in ts:
            mouse.Step()
            neuron.Step()
            V[k][t] = neuron.GetV()
            th[k][t] = neuron.GetThreshold()
            g1[k][t] = neuron.GetG1()
            g2[k][t] = neuron.GetG2()
            x[t],y[t] = mouse.GetPosition()
            I[t] = recf.GetSignal()

    return ts*dt,I,V,th,g1,g2,x,y

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Threshold decay simulation
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def ProbeThresholdDecay(Istim_duration,DeltaT_rec,nRecTimes,Istim=None,Iprobe=None,Iprobe_duration=None,use_findpeaks=True,findpeaks_args=None,thresholdmethod='model',**neuronArgs):
    """
    if Iprobe == None, performs experimental protocol similar to Anh-Tuan's
    otherwise, record the value of the model threshold variable
    """
    thresholdmethod = thresholdmethod.lower()
    if not ( thresholdmethod in ['model', 'maxcurvature', 'minderivative']):
        raise ValueError('thresholdmethod must be either model, maxcurvature, or minderivative')

    if has_no_threshold(neuronArgs['neuronType']) and (thresholdmethod == 'model'):
        print(' ... WARNING ::: the selected neuron %s has no explicit threshold, switching to minderivative method' % neuronArgs['neuronType'])
        thresholdmethod = 'minderivative'
    if thresholdmethod != 'model':
        use_findpeaks = True
    if (thresholdmethod != 'model') and (Iprobe is None):
        print(' ... WARNING ::: the selected neuron %s has no explicity threshold, threshold decay results in this script may be inaccurate'%neuronArgs['neuronType'])
        #Iprobe = neuronArgs['I0']

    dt = neuronArgs['dt']
    Istim = neuronArgs['I0'] if Istim is None else Istim

    # create noise
    noise = GetNoise(neuronArgs)
    if noise is None:
        neuronArgs['noiseSignal'] = lambda:0.0
    else:
        neuronArgs['noiseSignal'] = noise.GetSignal

    # create neuron
    neuron = GetNeuron(neuronArgs)
    synnoise = GetSynapticNoise(neuronArgs)
    if synnoise:
        neuron.AddInput(synnoise)
    
    # setup simulation
    ntrials = neuronArgs['ntrials']

    # total simulation time
    T = 1.05*Istim_duration + (nRecTimes+1) * DeltaT_rec # 5% more time than the stimulus duration + all the threshold probing intervals
    Tsteps = int(numpy.ceil(T / dt)) + 10
    ts = numpy.arange(Tsteps)

    # setup external current step
    t0 = numpy.ceil(0.05*Istim_duration / dt)
    Istim_duration = numpy.ceil(Istim_duration / dt)

    # creating output variables
    th_amp = numpy.zeros(nRecTimes)
    th2_amp = numpy.zeros(nRecTimes)
    DeltaT = numpy.arange(1,nRecTimes+1) * DeltaT_rec
    DeltaTexp = DeltaT.copy()

    # run :)
    if Iprobe is None: # there is no probing current, then the neuron itself should have an explicit threshold variable
        # creating data variables
        V = numpy.zeros(Tsteps)
        th = numpy.zeros(Tsteps)
        # setup external current step
        I = numpy.zeros(Tsteps)
        I[numpy.logical_and(ts >= t0, ts < (t0+Istim_duration))] = Istim
        # setup recording times
        DeltaT_rec_step = int(numpy.ceil(DeltaT_rec / dt))
        ts_rec = int(t0 + Istim_duration) + numpy.arange(1,nRecTimes+1) * DeltaT_rec_step
        theta0 = numpy.zeros(ntrials)
        for k in range(ntrials):
            # simulate the system
            neuron.Reset()
            for t in ts:
                neuron.Step(I[t])
                V[t] = neuron.GetV()
                th[t] = neuron.GetThreshold()
            th_idx = numpy.nonzero(V>(th+0.01))[0]
            if th_idx.size == 0:
                theta0[k] = numpy.nan
            else:
                theta0[k] = th[th_idx[0]-1]
            # record the threshold amplitude at each desired time step
            for i,tt in enumerate(ts_rec):
                th_amp[i] += th[tt]
                th2_amp[i] += th[tt]*th[tt]
            #print(theta0[k])
        theta0 = numpy.nanmean(theta0)
    else: # a probing current is delivered for each threshold estimation
        # if Iprobe_duration >= DeltaT_rec:
        #     raise ValueError('the duration of the probing current has be less that the recording interval')
        V = numpy.ones(Tsteps)*numpy.nan
        theta = numpy.ones(Tsteps)*numpy.nan
        #findpeaks_args = {'threshold':10} if findpeaks_args is None else findpeaks_args
        findpeaks_args = inp.set_default_kwargs(findpeaks_args,prominence=50.0)
        I = numpy.zeros(Tsteps)
        I[numpy.logical_and(ts >= t0, ts < (t0+Istim_duration))] = get_current_ramp(Istim,Istim_duration)
        ic = misc.get_empty_list(ntrials)
        tsInit = int(t0 + Istim_duration)
        theta0 = numpy.zeros(ntrials)
        # running ntrials initial stimulation to serve as IC for threshold probing
        for k in range(ntrials):
            neuron.Reset()
            for t in numpy.arange(tsInit):
                neuron.Step(I[t])
                V[t] = neuron.GetV()
                theta[t] = neuron.GetThreshold()
            ic[k] = neuron.GetStateAsIC()
            if use_findpeaks:
                if thresholdmethod == 'maxcurvature':
                    th,tspk0 = misc.calc_threshold_max_curvature(V,return_time_idx=True,**findpeaks_args)
                else: #elif thresholdmethod == 'minderivative':
                    th,tspk0 = misc.calc_threshold_min_derivative(V,return_time_idx=True,**findpeaks_args)
            else:
                tspk0 = misc.get_values_at_spike(numpy.arange(tsInit),V[:tsInit],Vth=59.0,tsStart=0,tsEnd=tsInit,tsDelay=-1,use_findpeaks=False,noSpkValue=None)
                th = theta[tspk0]
            theta0[k] = th if numpy.isscalar(th) else th[0]
        theta0 = numpy.nanmean(theta0)
        # running ntrials probes for the threshold for each delay after the main stimulation
        Iprobe_duration = int(numpy.ceil(Iprobe_duration / dt))
        # setup recording times
        DeltaT_rec_step = int(numpy.ceil(DeltaT_rec / dt))
        #t0 = (Iprobe_duration - DeltaT_rec_step)
        #if t0 > 0:
        #    DeltaT += t0*dt
        #else:
        #    t0 = 0
        DeltaTexp = DeltaT.copy()
        ts_rec = numpy.arange(1,nRecTimes+1) * DeltaT_rec_step # ts of the probe current beginning #= int(t0) + numpy.arange(1,nRecTimes+1) * DeltaT_rec_step
        ntrials_missed = numpy.zeros(nRecTimes)
        for j,tProbe in enumerate(ts_rec): # for each delay
            tsProbe = numpy.arange(tProbe+Iprobe_duration+10)#tsProbe = numpy.arange(tProbe+10)
            Iext = numpy.zeros(tProbe+Iprobe_duration+10)#Iext = numpy.zeros(tProbe+10)
            Vprobe = numpy.zeros(tProbe+Iprobe_duration+10)#Vprobe = numpy.zeros(tProbe+10)
            theta_probe = numpy.zeros(tProbe+Iprobe_duration+10)
            Iext[numpy.logical_and(tsProbe >= tProbe, tsProbe < (tProbe+Iprobe_duration))] = get_current_ramp(Iprobe,Iprobe_duration) #Iext[numpy.logical_and(tsProbe >= (tProbe-Iprobe_duration), tsProbe < tProbe)] = get_current_ramp(Iprobe,Iprobe_duration)
            tspk = 0.0
            for k in range(ntrials):
                neuron.Reset() # reset variables
                neuron.SetAttrib(**ic[k]) # setting the IC as the one calculated in the first loop
                for t in tsProbe: # running the simulation and recording the potential
                    neuron.Step(Iext[t])
                    Vprobe[t] = neuron.GetV()
                    theta_probe[t] = neuron.GetThreshold()
                if use_findpeaks:
                    th,tspk0 = misc.calc_threshold_max_curvature(Vprobe,return_time_idx=True,**findpeaks_args)
                    th = th[tspk0 > tProbe]
                    tspk0 = tspk0[tspk0 > tProbe]
                else:
                    tspk0 = misc.get_values_at_spike(tsProbe,Vprobe,Vth=59.0,tsStart=0,tsEnd=tsProbe[-1],tsDelay=-1,use_findpeaks=False,noSpkValue=None)
                    tspk0 = tspk0[tspk0 > tProbe]
                    th = theta_probe[tspk0]
                if not numpy.isscalar(th):
                    if th.size > 0:
                        th = th[0]
                        tspk0 = tspk0[0]
                    else:
                        th = numpy.nan
                        tspk0 = numpy.nan
                        ntrials_missed[j] += 1
                if not numpy.isnan(th):
                    th_amp[j] += th
                    th2_amp[j] += th*th
                    tspk += tspk0 * dt
            tspk = tspk / (ntrials-ntrials_missed[j]) if ntrials!=ntrials_missed[j] else numpy.nan
            if not numpy.isnan(tspk):
                DeltaT[j] = tspk if tspk > 0.0 else DeltaT[j]
            t_rec = tsInit+(j+1)*DeltaT_rec_step
            V[numpy.logical_and( ts>=t_rec,ts<(t_rec+DeltaT_rec_step+5) )] = Vprobe[-(DeltaT_rec_step+5):]
            theta[numpy.logical_and( ts>=t_rec,ts<(t_rec+DeltaT_rec_step+5) )] = theta_probe[-(DeltaT_rec_step+5):]
            I[numpy.logical_and( ts>=t_rec,ts<(t_rec+DeltaT_rec_step+5) )] = Iext[-(DeltaT_rec_step+5):]
        ntrials = ntrials - ntrials_missed
        ntrials[ntrials==0] = numpy.nan

    th_amp /= ntrials
    th_std = numpy.sqrt(th2_amp/ntrials - th_amp**2.0)

    if Iprobe is None:
        return DeltaT,th_amp,th_std,theta0
    else:
        return DeltaT,th_amp,th_std,ts*dt,I,V,tsInit*dt,DeltaTexp,theta0,theta

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Single Neuron with external stimulus
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def RunSingleNeuron(T,stimArgs,**simArgs):
    """
    Runs single neuron simulation;

    returns:
    V -> if stimArgs['I0'] is list, then V[i][j] -> i is the current index, and j is the trial index
         otherwise, returns V[j] -> j is the trial index;
         other neuron variables follow the same pattern
    I -> if stimArgs['I0'] is list, then I[i] -> is the stimulus current for each trial in V[i]
    t -> numpy.ndarray with the time steps
    
    parameters
    simArgs = dict(**neu.GetDefaultParamSet(paramset),dt=dt,noiseSignal=lambda:0.0,ntrials=ntrials)
    stimArgs = inp.get_input_stimulus_args()
    """
    if (type(stimArgs['I0']) is list) or (type(stimArgs['I0']) is numpy.ndarray):
        n = len(stimArgs['I0'])
        V = misc.get_empty_list(n)
        th = misc.get_empty_list(n)
        g1 = misc.get_empty_list(n)
        g2 = misc.get_empty_list(n)
        I = misc.get_empty_list(n)
        sArgs = stimArgs.copy()
        for i in range(n):
            sArgs['I0'] = stimArgs['I0'][i]
            t,I[i],V[i],th[i],g1[i],g2[i] = RunSingleNeuron(T,sArgs,**simArgs)
        return t,I,V,th,g1,g2
    else:
        dt = simArgs['dt']

        # create noise
        noise = GetNoise(simArgs)
        if noise is None:
            simArgs['noiseSignal'] = lambda:0.0
        else:
            simArgs['noiseSignal'] = noise.GetSignal

        # create neuron
        neuron = GetNeuron(simArgs)
        synnoise = GetSynapticNoise(simArgs)
        if synnoise:
            neuron.AddInput(synnoise)

        # setup simulation
        ntrials = simArgs['ntrials']
        Tsteps = int(numpy.ceil(T / dt))
        ts = numpy.arange(Tsteps)
        V = [numpy.zeros(Tsteps) for k in range(ntrials)]
        th = [numpy.zeros(Tsteps) for k in range(ntrials)]
        g1 = [numpy.zeros(Tsteps) for k in range(ntrials)]
        g2 = [numpy.zeros(Tsteps) for k in range(ntrials)]
        I = get_input_current(ts,dt,stimArgs)

        # run :)
        for k in range(ntrials):
            neuron.Reset()
            #print('-----------------')
            #print('trial = %d' % k)
            #print('I0  = %.2f' % stimArgs['I0'])
            #print('V0  = %.2f' % neuron.GetV())
            #print('th0 = %.2f' % neuron.GetThreshold())
            #print('g10 = %.2f' % neuron.GetG1())
            #print('g20 = %.2f' % neuron.GetG2())
            #print(' ')
            for t in ts:
                neuron.Step(I[t])
                V[k][t] = neuron.GetV()
                th[k][t] = neuron.GetThreshold()
                g1[k][t] = neuron.GetG1()
                g2[k][t] = neuron.GetG2()

        return ts*dt,I,V,th,g1,g2

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### AHP experiment
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def RunSimOverParamRange(par,parValues,T=100.0,**neuronArgs):
    # forces IC to be at the reset potential
    neuronArgs['VInit'] = neuronArgs['Vr']
    neuronArgs['gInit'] = neuronArgs['DeltaGK'] # assigned in the for loop
    neuronArgs['g2Init'] = neuronArgs['DeltaG2'] # assigned in the for loop
    neuronArgs['thetaInit'] = neuronArgs['theta0'] + neuronArgs['DeltaTheta']
    dt = neuronArgs['dt']
    Iext = neuronArgs['I0']
    neuron = GetNeuron(neuronArgs)
    nt = int(numpy.ceil(T/dt))
    ntrials = len(parValues)
    ts = numpy.arange(nt)
    V = [numpy.zeros(nt) for k in range(ntrials)]
    g1 = [numpy.zeros(nt) for k in range(ntrials)]
    g2 = [numpy.zeros(nt) for k in range(ntrials)]
    th = [numpy.zeros(nt) for k in range(ntrials)]
    for k,p in enumerate(parValues):
        neuron.Reset()
        if par == 'I0':
            Iext = p
        else:
            s = {par:p}
            if par == 'DeltaGK':
                s = {**s,'gK':p,'gInit':p}
            if par == 'DeltaG2':
                s = {**s,'g2':p,'g2Init':p}
            neuron.SetAttrib(**s)
        for t in ts:
            neuron.Step(Iext)
            V[k][t] = neuron.GetV()
            g1[k][t] = neuron.GetG1()
            g2[k][t] = neuron.GetG2()
            th[k][t] = neuron.GetThreshold()
    return ts*dt,V,g1,g2,th

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Analytic solution
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def GetLIFDTKAnalyticSolution(T,**neuronArgs):
    # I'm not sure the solution for V(t) is correct
    EK = neuronArgs['EK']
    dgk = neuronArgs['DeltaGK']
    VR = neuronArgs['Vr']
    Vb = neuronArgs['Vb']
    tau = neuronArgs['tau']
    R = neuronArgs['R']
    tauK = neuronArgs['tauK']
    Iext = neuronArgs['I0']
    dt = neuronArgs['dt'] * 1e-3 # miliseconds
    theta0 = neuronArgs['theta0']
    DeltaTheta = neuronArgs['DeltaTheta']
    tauTheta = neuronArgs['tauTheta']
    theta = lambda t: theta0 + DeltaTheta * numpy.exp(-t / tauTheta)
    gK = lambda t: dgk * numpy.exp(-t/tauK)
    h = lambda t: numpy.exp(( t-R*tauK*gK(t) )/tau)
    b = R*dgk*tauK/tau
    nt = int(numpy.ceil(T/dt))
    V = lambda t: EK + VR*h(0.0)/h(t) + (((tauK/tau)**(1.0+tauK/tau))*( (R*dgk)**(tauK/tau) )*(EK-Vb-R*Iext)/h(t)) * specf.Gammainc(-tauK/tau,b,R*tauK*gK(t)/tau)
    tt = numpy.arange(nt)*dt
    return tt,V(tt),theta(tt),gK(tt)

def IntegrateLIFDTKFromSpike(T,**neuronArgs):
    neuronArgs['VInit'] = neuronArgs['Vr']
    neuronArgs['gInit'] = neuronArgs['DeltaGK']
    neuronArgs['thetaInit'] = neuronArgs['theta0'] + neuronArgs['DeltaTheta']
    dt = neuronArgs['dt']
    Iext = neuronArgs['I0']
    neuron = GetNeuron(neuronArgs)
    nt = int(numpy.ceil(T/dt))
    t = numpy.arange(nt)*dt
    V = numpy.zeros(nt)
    th = numpy.zeros(nt)
    gK = numpy.zeros(nt)
    for k,tt in enumerate(t):
        neuron.Step(Iext)
        V[k] = neuron.GetV()
        th[k] = neuron.GetThreshold()
        gK[k] = neuron.GetG1()
    return t,V,th,gK


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Helper Functions
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def get_init_neuronargs(neuronArgs=None):
    neuronArgs = neuronArgs_default if neuronArgs is None else neuronArgs
    return [k for k in neuronArgs.keys() if 'Init' in k]

def replace_keys(d,old_keys,new_keys):
    c = d.copy()
    if len(new_keys) != len(old_keys):
        raise ValueError()
    for i,k in enumerate(old_keys):
        c[new_keys[i]] = c.pop(k,None)
    return c

def keep_keys(d,keys_to_keep):
    c = d.copy()
    for k in d:
        if k not in keys_to_keep:
            c.pop(k,None)
    return c

def get_func_param(f):
    f_code = f.__code__
    args = f_code.co_varnames[:f_code.co_argcount + f_code.co_kwonlyargcount]
    return [ a for a in args if a != 'self']

def get_current_ramp(Imax,nTimeSteps):
    nTimeSteps = numpy.abs(nTimeSteps)
    if nTimeSteps > 1:
        a = Imax / float(nTimeSteps)
    else:
        return Imax
    return numpy.asarray([ a*float(t+1) for t in range(int(nTimeSteps)) ])

def get_input_current(ts,dt,stimArgs):
    tIext = int(numpy.ceil(stimArgs['tStim'] / dt))
    dtIext = int(numpy.ceil(stimArgs['DeltaTStim'] / dt))
    I = numpy.zeros(len(ts))
    if stimArgs['stimtype'].lower() == 'step':
        I[numpy.logical_and(ts >= tIext, ts < (tIext+dtIext))] = stimArgs['I0']
    elif stimArgs['stimtype'].lower() == 'ramp':
        I[numpy.logical_and(ts >= tIext, ts < (tIext+dtIext))] = get_current_ramp(stimArgs['I0'],dtIext)
    return I

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Trajectory classes
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class Trajectory:
    def __init__(self,x0,y0,v0x,v0y,dt):
        self.x0 = x0
        self.y0 = y0
        self.x = x0
        self.y = y0
        self.vx = v0x
        self.vy = v0y
        self.v0x = v0x
        self.v0y = v0y
        self.dt = dt
        return
    def Step(self):
        return
    def GetPosition(self):
        return self.x,self.y
    def Reset(self):
        self.vx = self.v0x
        self.vy = self.v0y
        self.x = self.x0
        self.y = self.y0

class StraightTrajectory(Trajectory):
    def __init__(self,x0,y0,v0x,v0y,dt):
        Trajectory.__init__(self,x0,y0,v0x,v0y,dt)
    def Step(self):
        self.x = self.x + self.dt * self.vx;
        self.y = self.y + self.dt * self.vy;

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Synaptic input classes
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

def simulate_syn_poisson_noise(t,noiseParam,noiseClass=None):
    #t = numpy.linspace(0,Tmax,nT)
    if type(noiseClass) is type(None):
        noiseClass = SynapticPoissonNoise
    s = noiseClass(**keep_keys(noiseParam,get_func_param(noiseClass.__init__)))
    V = numpy.zeros(t.size)
    for i,tt in enumerate(t):
        V[i] = s.GetSignal()
    return V

class SynapticInput:
    def __init__(self):
        return
    def GetSignal(self):
        return 0.0

class SynapticWhiteNoise(SynapticInput):
    def __init__(self,stddev,dt,mean=0.0):
        stddev = stddev / 2.0
        self.sqrt_D_dt = numpy.sqrt(stddev*stddev*dt)
        self.mean = mean
    
    def GetSignal(self):
        return self.mean + self.sqrt_D_dt * numpy.random.normal()

class PoissonProcess(SynapticInput):
    def __init__(self,r,J=1.0):
        self.r = 1.0-numpy.exp(-r) # probability of firing is constant
        self.J = J
    def GetSignal(self):
        return self.J*float(numpy.random.random()<self.r)

class PoissonProcess_byInterval(SynapticInput):
    def __init__(self,r):
        self.r = r
        self.t = -1
        self.t0 = self.NextInterval()
    def GetSignal(self):
        return float(numpy.random.random()<self.r)
    def NextInterval(self):
        return 1 - int(numpy.round(numpy.log(numpy.random.random()) / self.r))
    
    def GetSignal(self):
        self.t += 1
        if self.t == self.t0:
            self.t0 += self.NextInterval()
            return 1.0
        return 0.0

class SynapticPoissonNoise(PoissonProcess):
    def __init__(self,poissonRate,tau1,tau2,J,p_ex,f0,dt):
        """
        A Kuva et al 2001 type synapse generated by a Poisson process of rate poissonRate

        poissonRate -> rate of the underlying poisson process (i.e., rate at which this process generates a PSP either excitatory or inhibitory)
        tau1 -> increase characteristic time of the PSP
        tau2 -> decrease characteristic time of the PSP
        J -> intensity of the PSP
        p_ex -> probability that the PSP will be excitatory (i.e., J>0 for a given PSP)
        f0 -> baseline value for the signal (FP for f == f0 * tau1; FP for g == f0)
        dt -> integration time step
        """
        PoissonProcess.__init__(self,poissonRate)
        self.invTau1 = 1.0/tau1
        self.invTau2 = 1.0/tau2
        self.J = J
        self.p_ex = p_ex
        self.f0 = f0
        self.dt = dt
        self.f = tau1*f0 #0.0
        self.g = f0 #0.0
        self.df1 = 0.0
        self.df2 = 0.0
        self.dg1 = 0.0
        self.dg2 = 0.0
        self.f1 = 0.0
        self.g1 = 0.0
    
    def GetSignal(self):
        # the underlying posisson process
        x = PoissonProcess.GetSignal(self)

        # the sign of the interaction is random, excitatory with probability p_ex
        s = 2.0*float(numpy.random.random() < self.p_ex)-1.0

        # integrating the process (RK2)
        self.df1 = self.g - self.f * self.invTau1
        self.dg1 = s*self.J * x + (self.f0 - self.g) * self.invTau2 
        self.f1 = self.f + self.dt * self.df1
        self.g1 = self.g + self.dt * self.dg1
        self.df2 = self.g1 - self.f1 * self.invTau1
        self.dg2 = s*self.J * x + (self.f0 - self.g1) * self.invTau2
        self.f = self.f + (self.df1 + self.df2) * self.dt / 2.0
        self.g = self.g + (self.dg1 + self.dg2) * self.dt / 2.0
        return self.f #+ self.f0

class ReceptiveField(SynapticInput):
    def __init__(self,I0,x0,y0,radius):
        self.R = radius
        self.x0 = x0
        self.y0 = y0
        self.I0 = I0
        self.traj = None

    def SetTrajectory(self,trajectory):
        self.traj = trajectory

    def GetSignal(self):
        x,y = self.traj.GetPosition()
        # if the trajectory is inside this receptive field, return the constant current I0, otherwise return 0 current
        if ((x - self.x0)*(x - self.x0) + (y - self.y0)*(y - self.y0)) < (self.R*self.R):
            return self.I0
        return 0.0
    
    def Reset(self):
        return

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIF neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIF:
    def __init__(self,dt,VInit,Vr,Vb,tau,R,theta,noiseSignal,setIC = True):
        self.Vrec = VInit
        self.VInit = VInit
        self.Vr = Vr
        self.Vb = Vb
        self.theta = theta
        self.invTau = 1.0 / tau
        self.R = R
        self.noiseSignal = noiseSignal
        self.dV1 = 0.0
        self.dV2 = 0.0
        self.dt = dt
        self.input = []
        if setIC:
            self.SetIC(VInit=VInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.R*Iext - self.V + self.Vb)*self.invTau
        self.dV2 = (self.R*Iext - (self.V + self.dt*self.dV1 + self.noiseSignal()) + self.Vb)*self.invTau
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetV(self):
        return self.Vrec
    
    def GetThreshold(self):
        return self.theta
    
    def GetGK(self):
        return 0.0
    
    def GetG1(self):
        return self.GetGK()

    def GetG2(self):
        return 0.0
    
    def AddInput(self,inp):
        self.input.append(inp)
    
    def SumInput(self):
        s = 0.0
        for I in self.input:
            s += I.GetSignal()
        return s
    
    def Reset(self):
        self.V = self.VInit
        self.Vrec = self.V

    def SetAttrib(self,**kwargs):
        self.__dict__.update(kwargs)
        if any(['init' in s.lower() for s in kwargs.keys()]):
            self.Reset()

    def GetStateAsIC(self):
        return dict(VInit=self.V)
    
    def GetRestingState(self,Iext=0.0):
        return dict(V=self.Vb+self.R*Iext)
    
    def SetIC(self,**kwargs):
        if any([numpy.isnan(v) for v in kwargs.values()]):
            d = self.GetRestingState()
            self.SetAttrib(**dict([ ( (k[0] if k == 'gK' else k)+'Init',v) for k,v in d.items() ]))
        else:
            self.SetAttrib(**dict([ ( (k[0] if k == 'gK' else k)+('' if 'init' in k.lower() else 'Init' ),v) for k,v in kwargs.items() ]))

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDT neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDT(LIF):
    def __init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC = True):
        LIF.__init__(self,dt,VInit,Vr,Vb,tau,R,thetaInit,noiseSignal,setIC=False)
        self.thetaInit = thetaInit
        self.theta0 = theta0
        self.DeltaTheta = DeltaTheta
        self.invTauTheta = 1.0 / tauTheta
        self.dth1 = 0.0
        self.dth2 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.R*Iext - self.V + self.Vb)*self.invTau
        self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.dV2 = (self.R*Iext - (self.V + self.dt*self.dV1 + self.noiseSignal()) + self.Vb)*self.invTau
        self.dth2 = (self.theta0 - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.theta += self.DeltaTheta
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.Vrec = self.V

    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta)

    def GetRestingState(self,Iext=0.0):
        return dict(V=self.Vb+self.R*Iext,theta=self.theta0)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### QIF neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class QIF(LIF):
    def __init__(self,dt,VInit,Vr,Vb,Vc,tau,R,theta,noiseSignal,setIC=True):
        LIF.__init__(self,dt,VInit,Vr,Vb,tau,R,theta,noiseSignal,setIC=False)
        self.V1 = VInit
        self.Vc = Vc
        if setIC:
            self.SetIC(VInit=VInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = self.R*Iext + (self.V-self.Vc) * (self.V-self.Vb) * self.invTau
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.dV2 = self.R*Iext + (self.V1-self.Vc) * (self.V1-self.Vb) * self.invTau
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetV(self):
        return self.Vrec
    
    def GetThreshold(self):
        return self.theta
    
    def AddInput(self,inp):
        self.input.append(inp)
    
    def SumInput(self):
        s = 0.0
        for i in range(len(self.input)):
            s += self.input[i].GetSignal()
        return s
    
    def GetRestingState(self,Iext=0.0):
        s2 = self.invTau * (self.invTau*(self.Vb-self.Vc)**2 - 4*self.R*Iext )
        if s2 < 0:
            Vs1 = numpy.nan
            Vs2 = Vs1
        else:
            Vs = (self.Vb+self.Vc)/2.0
            Vs1 = Vs + numpy.sqrt(s2)/(2.0*self.invTau)
            Vs2 = Vs - numpy.sqrt(s2)/(2.0*self.invTau)
        return dict(V=numpy.min([Vs1,Vs2]))

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### QIFDT neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class QIFDT(QIF):
    def __init__(self,dt,VInit,thetaInit,Vr,Vb,Vc,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC=True):
        QIF.__init__(self,dt,VInit,Vr,Vb,Vc,tau,R,thetaInit,noiseSignal,setIC=False)
        self.thetaInit = thetaInit
        self.theta0 = theta0
        self.DeltaTheta = DeltaTheta
        self.invTauTheta = 1.0 / tauTheta
        self.dth1 = 0.0
        self.dth2 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = self.R*Iext + (self.V-self.Vc) * (self.V-self.Vb) * self.invTau
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.dV2 = self.R*Iext + (self.V1-self.Vc) * (self.V1-self.Vb) * self.invTau
        self.dth2 = (self.theta0 - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.theta += self.DeltaTheta
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.Vrec = self.V

    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta)

    def GetRestingState(self,Iext=0.0):
        s2 = self.invTau * (self.invTau*(self.Vb-self.Vc)**2 - 4*self.R*Iext )
        if s2 < 0:
            Vs1 = numpy.nan
            Vs2 = Vs1
        else:
            Vs = (self.Vb+self.Vc)/2.0
            Vs1 = Vs + numpy.sqrt(s2)/(2.0*self.invTau)
            Vs2 = Vs - numpy.sqrt(s2)/(2.0*self.invTau)
        return dict(V=numpy.min([Vs1,Vs2]),theta=self.theta0)
    
"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDTK neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDTK(LIFDT):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,noiseSignal,setIC=True):
        LIFDT.__init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC=False)
        self.gInit = gInit
        self.EK = EK
        self.DeltaGK = DeltaGK
        self.invTauK = 1.0 / tauK
        self.gK = gInit
        self.dg1 = 0.0
        self.dg2 = 0.0
        self.V1 = 0.0
        self.gK1 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.R*Iext - self.R*self.gK*(self.V-self.EK))*self.invTau
        self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        self.dV2 = (self.Vb - self.V1 + self.R*Iext - self.R*self.gK1*(self.V1-self.EK))*self.invTau
        self.dth2 = (self.theta0 - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.theta += self.DeltaTheta
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetGK(self):
        return self.gK

    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.gK = self.gInit
        self.Vrec = self.V

    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta,gInit=self.gK)

    def GetRestingState(self,Iext=0.0):
        return dict(V=self.Vb+self.R*Iext,theta=self.theta0,gK=0.0)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDT2K neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDT2K(LIFDTK):
    def __init__(self,dt,VInit,thetaInit,gInit,g2Init,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,E2,tau2,DeltaG2,noiseSignal,setIC=True):
        LIFDTK.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,noiseSignal,setIC=False)
        self.g2Init = g2Init
        self.E2 = E2
        self.DeltaG2 = DeltaG2
        self.invTau2 = 1.0 / tau2
        self.g2 = g2Init
        self.dg21 = 0.0
        self.dg22 = 0.0
        self.gK1 = 0.0
        self.g21 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit,g2Init=g2Init)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.R*Iext - self.R*self.gK*(self.V-self.EK) - self.R*self.g2*(self.V-self.E2))*self.invTau
        self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.dg21 = -self.g2 * self.invTau2
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        self.g21 = self.g2 + self.dt * self.dg21
        self.dV2 = (self.Vb - self.V1 + self.R*Iext - self.R*self.gK1*(self.V1-self.EK) - self.R*self.g21*(self.V1-self.E2))*self.invTau
        self.dth2 = (self.theta0 - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.dg22 = -self.g21 * self.invTau2
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0
        self.g2 = self.g2 + (self.dg21 + self.dg22) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.theta += self.DeltaTheta
            self.gK += self.DeltaGK
            self.g2 += self.DeltaG2
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetG2(self):
        return self.g2

    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.gK = self.gInit
        self.g2 = self.g2Init
        self.Vrec = self.V
    
    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta,gInit=self.gK,g2Init=self.g2)
    
    def GetRestingState(self,Iext=0.0):
        return dict(V=self.Vb+self.R*Iext,theta=self.theta0,gK=0.0,g2=0.0)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDTKz neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDTKz(LIFDTK):
    def __init__(self,dt,VInit,thetaInit,gInit,g2Init,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,E2,tau2,G2,noiseSignal,setIC=True):
        LIFDTK.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,noiseSignal,setIC=False)
        self.g2Init = g2Init
        self.E2 = E2
        self.G2 = G2
        self.invTau2 = 1.0 / tau2
        self.gK1 = 0.0
        self.g2 = g2Init
        self.dg21 = 0.0
        self.dg22 = 0.0
        self.g21 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit,g2Init=g2Init)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.R*Iext - self.R*self.gK*(self.V-self.EK) + self.R*self.g2)*self.invTau #*(self.V-self.E2)
        self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.dg21 = (-self.G2*(self.V-self.E2)-self.g2) * self.invTau2
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        self.g21 = self.g2 + self.dt * self.dg21
        self.dV2 = (self.Vb - self.V1 + self.R*Iext - self.R*self.gK1*(self.V1-self.EK) + self.R*self.g21)*self.invTau #*(self.V1-self.E2)
        self.dth2 = (self.theta0 - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.dg22 = (-self.G2*(self.V1-self.E2)-self.g21) * self.invTau2
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0
        self.g2 = self.g2 + (self.dg21 + self.dg22) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.theta += self.DeltaTheta
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

    def GetG2(self):
        return self.g2

    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.gK = self.gInit
        self.g2 = self.g2Init
        self.Vrec = self.V

    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta,gInit=self.gK,g2Init=self.g2)

    def GetRestingState(self,Iext=0.0):
        Vs = (self.Vb + self.R*Iext-self.R*self.G2*self.E2)/(1-self.R*self.G2)
        g2s = self.G2*(Vs-self.E2)
        return dict(V=Vs,theta=self.theta0,gK=0.0,g2=g2s)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDTBoundedK neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDTBoundedK(LIFDTK):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,noiseSignal,setIC=True):
        LIFDTK.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,noiseSignal,setIC=False)
        self.thetaMax = thetaMax
        self.normDeltaTheta = DeltaTheta / (thetaMax - theta0) # normalizing constant for theta increment
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit)

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.theta += self.normDeltaTheta * (self.thetaMax - self.theta) # limits the growth of theta up to thetaMax
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDTKBounded neuron
### 
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDTKBounded(LIFDTK):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,gMax,EK,tauK,DeltaGK,noiseSignal,setIC=True):
        LIFDTK.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,noiseSignal,setIC=False)
        self.gMax = gMax
        self.normDeltaGK = DeltaGK / gMax # normalizing constant for gk increment
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit)

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.theta += self.DeltaTheta
            self.gK += self.normDeltaGK * (self.gMax - self.gK) # limits the growth of gk up to gMax
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Exponential LIF neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFex(LIF):
    def __init__(self,dt,VInit,Vr,Vb,tau,R,theta,DeltaT,noiseSignal,Vpeak=None,setIC=True):
        LIF.__init__(self,dt,VInit,Vr,Vb,tau,R,theta,noiseSignal,setIC=False)
        self.DeltaT = DeltaT
        self.V1 = 0.0
        if Vpeak is None:
            Vs = self.CalcRealThreshold()
            self.Vpeak = Vs + numpy.abs(self.theta - Vs)
            print('exponential threshold = %f' % self.Vpeak)
        else:
            self.Vpeak = Vpeak
        if setIC:
            self.SetIC(VInit=VInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.theta)/self.DeltaT) + self.R*Iext)*self.invTau
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.theta)/self.DeltaT) + self.R*Iext)*self.invTau
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
    
    def Step(self,Iext = 0.0):
        if self.V > self.Vpeak:
            self.V = self.Vr
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def CalcRealThreshold(self):
        return calc_real_exp_threshold(self.Vb,self.DeltaT,self.theta)
    
    def GetRestingState(self,Iext=0.0):
        return dict(V=calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,self.theta,returnRest=True))

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Adaptive Exponential LIF neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFAdEx(LIFex):
    def __init__(self,dt,wInit,VInit,Vr,Vb,tau,R,theta,DeltaT,gW,tauW,DeltaW,noiseSignal,Vpeak=None,setIC=True):
        LIFex.__init__(self,dt,VInit,Vr,Vb,tau,R,theta,DeltaT,noiseSignal,Vpeak=Vpeak,setIC=False)
        self.wInit = wInit
        self.dw1 = 0.0
        self.dw2 = 0.0
        self.DeltaW = DeltaW
        self.invTauW = 1.0 / tauW
        self.gW = gW
        self.w1 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,wInit=wInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.theta)/self.DeltaT) + self.R*(Iext - self.w))*self.invTau
        self.dw1 = (self.gW * (self.V - self.Vb) - self.w) * self.invTauW
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.w1 = self.w + self.dt*self.dw1
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.theta)/self.DeltaT) + self.R*(Iext - self.w1))*self.invTau
        self.dw2 = (self.gW * (self.V1 - self.Vb) - self.w1) * self.invTauW
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.w = self.w + (self.dw1 + self.dw2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > self.Vpeak:
            self.V = self.Vr
            self.w += self.DeltaW
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetG2(self):
        return self.w
    
    def Reset(self):
        self.V = self.VInit
        self.w = self.wInit
        self.Vrec = self.V
    
    def GetStateAsIC(self):
        return {'VInit':self.V,'wInit':self.w}
    
    def GetRestingState(self,Iext=0.0):
        Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,self.theta,returnRest=True,Rgw=self.R*self.gW)
        ws = self.gW * (Vs - self.Vb)
        return dict(V=Vs,w=ws)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### inactivating slow-fast Exponential LIF neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFiEIFsf(LIF):
    def __init__(self,dt,VInit,hfInit,hsInit,Vr,Vb,tau,R,gNa,ENa,ka,Vc,ki,Vi,tauf,taus,noiseSignal,setIC=True):
        LIF.__init__(self,dt,VInit,Vr,Vb,tau,R,0.0,noiseSignal,setIC=False)
        self.hfInit = hfInit
        self.hsInit = hsInit
        self.hf = self.hfInit
        self.hs = self.hsInit
        self.V1 = 0.0
        self.dhf1 = 0.0
        self.dhs1 = 0.0
        self.dhf2 = 0.0
        self.dhs2 = 0.0
        self.hf1 = 0.0
        self.hs1 = 0.0
        self.gL = 1.0 / R
        self.ka = ka
        self.ki = ki
        self.Vi = Vi
        self.invTaus = 1.0 / taus
        self.invTauf = 1.0 / tauf
        self.hinf = iEIF_hinf
        self.VT = iEIF_VTfn(self.gL,ka,Vc,gNa,ENa)
        self.thetafn = iEIF_thetafn
        self.theta = self.thetafn(self.VT,self.ka,self.hs,self.hf)
        if setIC:
            self.SetIC(VInit=VInit,hfInit=hfInit,hsInit=hsInit)
        #print('VT = %f' % self.VT)
        #print('theta = %f' % self.theta)
    
    def dVFunc(self,V,hf,hs,Iext):
        return (hf*hs*my_exp((V-self.VT)/self.ka) + self.Vb - V + self.R*Iext) * self.invTau
    
    def dhFunc(self,V,h,invTa):
        return (self.hinf(V,self.Vi,self.ki) - h) * invTa

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = self.dVFunc(self.V,self.hf,self.hs,Iext)
        self.dhf1 = self.dhFunc(self.V,self.hf,self.invTauf)
        self.dhs1 = self.dhFunc(self.V,self.hs,self.invTaus)
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.hf1 = self.hf + self.dt*self.dhf1
        self.hs1 = self.hs + self.dt*self.dhs1
        self.dV2 = self.dVFunc(self.V1,self.hf1,self.hs1,Iext)
        self.dhf2 = self.dhFunc(self.V1,self.hf1,self.invTauf)
        self.dhs2 = self.dhFunc(self.V1,self.hs1,self.invTaus)
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.hf = self.hf + (self.dhf1 + self.dhf2) * self.dt / 2.0
        self.hs = self.hs + (self.dhs1 + self.dhs2) * self.dt / 2.0
        self.theta = self.thetafn(self.VT,self.ka,self.hs,self.hf)

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetG1(self):
        return self.hf

    def GetG2(self):
        return self.hs

    def Reset(self):
        self.V = self.VInit
        self.hf = self.hfInit
        self.hs = self.hsInit
        self.theta = self.thetafn(self.VT,self.ka,self.hs,self.hf)
    
    def GetStateAsIC(self):
        return {'VInit':self.V,'hfInit':self.hf,'hsInit':self.hs,'theta':self.theta}
    
    def GetRestingState(self,Iext=0.0):
        return dict(V=numpy.nan,hf=numpy.nan,hs=numpy.nan,theta=numpy.nan)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### inactivating Exponential LIF neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFiEIF(LIF):
    def __init__(self,dt,VInit,hsInit,Vr,Vb,tau,R,gNa,ENa,ka,Vc,ki,Vi,taus,noiseSignal,setIC=True):
        LIF.__init__(self,dt,VInit,Vr,Vb,tau,R,0.0,noiseSignal,setIC=False)
        self.hInit = hsInit
        self.h = self.hInit
        self.V1 = 0.0
        self.dh1 = 0.0
        self.dh2 = 0.0
        self.h1 = 0.0
        self.gL = 1.0 / R
        self.ka = ka
        self.ki = ki
        self.Vi = Vi
        self.invTauh = 1.0 / taus
        self.hinf = iEIF_hinf
        self.VT = iEIF_VTfn(self.gL,ka,Vc,gNa,ENa)
        self.thetafn = iEIF_thetafn
        self.theta = self.thetafn(self.VT,self.ka,self.h,1.0)
        #print('thetaInit = %f'%self.theta)
        if setIC:
            self.SetIC(VInit=VInit,hInit=hsInit)
        #print('VT = %f' % self.VT)
        #print('theta = %f' % self.theta)
    
    def dVFunc(self,V,h,Iext):
        return (h*my_exp((V-self.VT)/self.ka) + self.Vb - V + self.R*Iext) * self.invTau
    
    def dhFunc(self,V,h,invTa):
        return (self.hinf(V,self.Vi,self.ki) - h) * invTa

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = self.dVFunc(self.V,self.h,Iext)
        self.dh1 = self.dhFunc(self.V,self.h,self.invTauh)
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.h1 = self.h + self.dt*self.dh1
        self.dV2 = self.dVFunc(self.V1,self.h1,Iext)
        self.dh2 = self.dhFunc(self.V1,self.h1,self.invTauh)
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.h = self.h + (self.dh1 + self.dh2) * self.dt / 2.0
        self.theta = self.thetafn(self.VT,self.ka,self.h,1.0)

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vr
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetG2(self):
        return self.h
    
    def Reset(self):
        self.V = self.VInit
        self.h = self.hInit
        self.theta = self.thetafn(self.VT,self.ka,self.h,1.0)
    
    def GetStateAsIC(self):
        return {'VInit':self.V,'hInit':self.h,'theta':self.theta}
    
    def GetRestingState(self,Iext=0.0):
        return dict(V=numpy.nan,h=numpy.nan,theta=numpy.nan)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDTBoundedKLR neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDTBoundedKLR(LIFDTK):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,noiseSignal,setIC=True):
        LIFDTK.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,noiseSignal,setIC=False)
        self.rv = rv
        self.deltaV = deltaV
        self.thetaMax = thetaMax
        self.normDeltaTheta = DeltaTheta / (thetaMax - theta0) # normalizing constant for theta increment
        if setIC:
            self.SetIC(VInit=VInit,gInit=gInit,thetaInit=thetaInit)

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            #self.V = self.Vr + self.rv * (self.V - self.Vr) - self.deltaV
            self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV
            self.theta += self.normDeltaTheta * (self.thetaMax - self.theta) # limits the growth of theta up to thetaMax
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDTVBoundedKLR neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDTVBoundedKLR(LIFDTK):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,Rth,tauThetaV,noiseSignal,setIC=True):
        LIFDTK.__init__(self,dt,VInit,0.0,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,EK,tauK,DeltaGK,noiseSignal,setIC=False)
        self.rv = rv
        self.deltaV = deltaV
        self.thetaMax = thetaMax
        self.normDeltaTheta = DeltaTheta / (thetaMax - theta0) # normalizing constant for theta increment
        self.av = Rth / self.R # Rth is defined from: theta0(Iext) = Rth * Iext + theta0, but Iext=V/R, such that theta0(V) = (Rth/R) * V + theta0
        self.bv = 1.0/tauThetaV
        self.dthv1 = 0.0
        self.dthv2 = 0.0
        self.thetavInit = self.av * (self.V - self.Vb)
        self.thetav = self.thetavInit
        self.totalTheta = self.theta + self.thetav + self.theta0 # actual threshold for this model
        self.gK1 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,gInit=gInit,thetaInit=0.0)
        #            spk comp.  + volt comp.  + rest threshold
    
    #def cuberoot(self,x):
    #    return 0.0 if x == 0.0 else (abs(x)/x)*abs(x)**(1.0/3.0)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.R*Iext - self.R*self.gK*(self.V-self.EK))*self.invTau
        self.dth1 = -self.theta * self.invTauTheta
        self.dthv1 = self.av * (self.V - self.Vb) - self.bv * self.thetav#self.dthv1 = self.av * self.cuberoot(self.V - self.Vb) - self.bv * self.thetav
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        self.dV2 = (self.Vb - self.V1 + self.R*Iext - self.R*self.gK1*(self.V1-self.EK))*self.invTau
        self.dth2 = -(self.theta + self.dt * self.dth1) * self.invTauTheta
        self.dthv2 = self.av * (self.V1 - self.Vb) - self.bv * (self.thetav + self.dt * self.dthv1)#self.dthv2 = self.av * self.cuberoot(self.V1 - self.Vb) - self.bv * (self.thetav + self.dt * self.dthv1)
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.thetav = self.thetav + (self.dthv1 + self.dthv2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0
        self.totalTheta = self.theta + self.thetav + self.theta0

    def Step(self,Iext = 0.0):
        #print('th = {:g};    thv = {:g};    th0 = {:g};    total = {:g}'.format(self.theta,self.thetav,self.theta0,self.totalTheta))
        #wait = input("PRESS ENTER TO CONTINUE.")
        if self.V > self.totalTheta:
            #self.V = self.Vr + self.rv * (self.V - self.Vr) - self.deltaV
            self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV
            self.theta += self.normDeltaTheta * (self.thetaMax - self.totalTheta) # limits the growth of theta up to thetaMax
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetThreshold(self):
        return self.totalTheta

    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.thetav = self.thetavInit
        self.totalTheta = self.theta + self.thetav + self.theta0
        self.gK = self.gInit
        self.Vrec = self.V

    def GetStateAsIC(self):
        return {'VInit':self.V,'thetaInit':0.0,'thetavInit':self.av * (self.V - self.Vb),'gInit':self.gK}

    def GetRestingState(self,Iext=0.0):
        return dict(V=self.Vb+self.R*Iext,gK=0.0,thetav=0.0,theta=0.0)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDLTBoundedKLR neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDLTBoundedKLR(LIFDTBoundedKLR):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,Rth,noiseSignal,setIC=True):
        LIFDTBoundedKLR.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,noiseSignal,setIC=False)
        #self.gth = Rth #/ self.R # Rth is defined from: theta0(Iext) = Rth * Iext + theta0, but Iext=V/R, such that theta0(V) = (Rth/R) * V + theta0
        self.gth = Rth / self.R # Rth is defined from: theta0(Iext) = Rth * Iext + theta0, but Iext=V/R, such that theta0(V) = (Rth/R) * V + theta0
        #self.Vc = Vc # potential in which the linear dependence of theta0 ends
        #self.Vo = Vo # potential in which the linear dependence of theta0 starts
        #self.normDeltaTheta = DeltaTheta / (-60 - theta0) # normalizing constant for theta increment
        #self.Vspk = (self.gth*self.Vb-self.theta0)/(self.gth-1.0)
        #print(self.Vspk)
        if setIC:
            self.SetIC(VInit=VInit,gInit=gInit,thetaInit=thetaInit)

    def Theta(self,x):
        return float(x>0.0)
    
    def Plateau(self,x,a,b):
        return float(x>a and x<b)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.R*Iext - self.R*self.gK*(self.V-self.EK))*self.invTau
        self.dth1 = (self.theta0 + self.gth * (self.V-self.Vb)*(1.0-self.Theta(self.gK-1e-10)) - self.theta) * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        self.dV2 = (self.Vb - self.V1 + self.R*Iext - self.R*self.gK1*(self.V1-self.EK))*self.invTau
        self.dth2 = (self.theta0 + self.gth * (self.V1-self.Vb)*(1.0-self.Theta(self.gK1-1e-10)) - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            #self.V = self.Vr + self.rv * (self.V - self.Vr) - self.deltaV
            self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV
            #self.theta -= self.normDeltaTheta * (-60-self.theta) # limits the growth of theta up to thetaMax
            self.theta += self.normDeltaTheta * (self.thetaMax - self.theta) # limits the growth of theta up to thetaMax
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetRestingState(self,Iext=0.0):
        return dict(V=self.Vb+self.R*Iext,gK=0.0,theta=self.theta0)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDLTBoundedKLRIA neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDLTBoundedKLRIA(LIFDLTBoundedKLR):
    def __init__(self,dt,VInit,thetaInit,gInit,mIAInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,Rth,gIA,tauIA,VIA,kIA,noiseSignal,setIC=True):
        LIFDLTBoundedKLR.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,Rth,noiseSignal,setIC=False)
        self.mIAInit = mIAInit
        self.mIA = mIAInit
        self.gIA = gIA
        self.invTauIA = 1.0/tauIA
        self.VIA = VIA
        self.invkIA = 1.0/kIA
        self.dmIA1 = 0.0
        self.dmIA2 = 0.0
        self.mIA1 = 0.0
        self.Vrest = self.calcVrest()
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit,mIAInit=mIAInit)
        
    def Theta(self,x):
        return float(x>0.0)
    
    def Logistic(self,x):
        return 0.5+x/(2.0+2.0*abs(x)) # logistic function (a smooth sigmoid) between 0 (x<0) and 1 (x>0)
    
    def Logistic_numpy(self,x):
        return 0.5+x/(2.0+2.0*numpy.abs(x)) # logistic function (a smooth sigmoid) between 0 (x<0) and 1 (x>0)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.R*Iext - self.R*self.gK*(self.V-self.EK) - self.R*self.gIA*self.mIA*(1.0-self.Theta(self.gK-1e-10))*(self.V-self.EK))*self.invTau
        self.dth1 = (self.theta0 + self.gth * (self.V-self.Vrest)*(1.0-self.Theta(self.gK-1e-10)) - self.theta) * self.invTauTheta
        #self.dth1 = (self.theta0 + self.gth * (self.V+80.0629)*(1.0-self.Theta(self.gK-1e-10)) - self.theta) * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.dmIA1 = (self.Logistic((self.V - self.VIA)*self.invkIA) - self.mIA)*self.invTauIA
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        self.mIA1 = self.mIA + self.dt * self.dmIA1
        self.dV2 = (self.Vb - self.V1 + self.R*Iext - self.R*self.gK1*(self.V1-self.EK) - self.R*self.gIA*self.mIA1*(1.0-self.Theta(self.gK1-1e-10))*(self.V1-self.EK))*self.invTau
        self.dth2 = (self.theta0 + self.gth * (self.V1-self.Vrest)*(1.0-self.Theta(self.gK1-1e-10)) - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        #self.dth2 = (self.theta0 + self.gth * (self.V1+80.0629)*(1.0-self.Theta(self.gK1-1e-10)) - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.dmIA2 = (self.Logistic((self.V1 - self.VIA)*self.invkIA) - self.mIA1)*self.invTauIA
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0
        self.mIA = self.mIA + (self.dmIA1 + self.dmIA2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > self.theta:
            self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV
            self.theta += self.normDeltaTheta * (self.thetaMax - self.theta) # limits the growth of theta up to thetaMax
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def GetG2(self):
        return self.mIA

    def calcVrest(self):
        VV = self.GetRestingState()['V']
        if type(VV) is numpy.ndarray:
            return VV[0]
        else:
            return VV
    
    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.gK = self.gInit
        self.mIA = self.mIAInit
        self.Vrest = self.calcVrest()
        self.Vrec = self.V

    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta,gInit=self.gK,mIAInit=self.mIA)

    def GetRestingState(self,Iext=0.0):
        f = lambda s: 1.0/(4.0 * s + 2 * self.gIA*self.R*(1.0 + s))
        a = lambda s: self.VIA*self.gIA*self.R+self.EK*self.gIA*self.R+(-1)*(1.0/self.invkIA)*(2+self.gIA*self.R)+2*self.VIA*s+2*(self.Vb+self.R*Iext)*s+self.VIA*self.gIA*self.R*s+self.EK*self.gIA*self.R*s
        b2 = lambda s: ((self.VIA+self.EK)*self.gIA*self.R+(-1)*(1.0/self.invkIA)*(2+self.gIA*self.R)+2*(self.VIA+(self.Vb+self.R*Iext))*s+(self.VIA+self.EK)*self.gIA*self.R*s)**2+(-4)*(2*s+self.gIA*self.R*(1+s))*(self.VIA*self.EK*self.gIA*self.R+(-1)*(2*(self.Vb+self.R*Iext)+self.EK*self.gIA*self.R)*((1.0/self.invkIA)+(-1)*self.VIA*s))
        ss = numpy.asarray([-1,1,-1,1])
        rr = numpy.asarray([-1,-1,1,1])
        bb = b2(ss)
        if numpy.all(bb<0):
            Vs = numpy.nan
            gKs = numpy.nan
            thetas = numpy.nan
            mIAs = numpy.nan
        else:
            VV = lambda s1,r1: f(s1)*(   a(s1)   +   r1*numpy.sqrt(bb)   )
            Vs = VV(ss,rr)
            VTrue = numpy.logical_and(numpy.logical_or(numpy.logical_and(Vs > self.VIA,ss>0),numpy.logical_and(Vs < self.VIA,ss<0)),bb>=0.0)
            Vs = Vs[VTrue]
            gKs = 0.0*Vs
            thetas = self.theta0*numpy.ones(Vs.shape)
            mIAs = self.Logistic_numpy((Vs - self.VIA)*self.invkIA)
            #if Vs.size == 1:
            Vs = Vs[0]
            gKs = gKs[0]
            thetas = thetas[0]
            mIAs = mIAs[0]
        #print(dict(V=Vs,gK=gKs,theta=thetas,mIA=mIAs))
        return dict(V=Vs,gK=gKs,theta=thetas,mIA=mIAs)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### LIFDLT neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class LIFDLT(LIFDT):
    def __init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,Rth,noiseSignal,setIC = True):
        LIFDT.__init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC = False)
        self.Rth = Rth
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.R*Iext - self.V + self.Vb)*self.invTau
        self.dth1 = (self.theta0 + self.Rth*Iext - self.theta) * self.invTauTheta
        self.dV2 = (self.R*Iext - (self.V + self.dt*self.dV1 + self.noiseSignal()) + self.Vb)*self.invTau
        self.dth2 = (self.theta0 + self.Rth*Iext - (self.theta + self.dt * self.dth1)) * self.invTauTheta
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0

    def GetRestingState(self,Iext=0.0):
        return dict(V=self.Vb+self.R*Iext,theta=self.theta0+self.Rth*Iext)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### EIFHDTBoundSigKLR neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class EIFHDTBoundSigKLR(LIFDTBoundedKLR):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,DeltaT,VT,lf_s,lf_I0,Rth,tauRiseTheta,thetasInit,noiseSignal,setIC=True):
        LIFDTBoundedKLR.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,noiseSignal,setIC=False)
        self.lf_s = lf_s
        self.lf_I0 = lf_I0
        self.RTheta = Rth
        self.DeltaT = DeltaT
        self.VT = VT
        self.thetasInit = thetasInit
        self.thetas = thetasInit
        self.invTauRiseTheta = 1.0/tauRiseTheta
        self.dths1 = 0.0
        self.dths2 = 0.0
        self.normDeltaTheta = DeltaTheta / thetaMax # normalizing constant for theta increment
        self.thetaRest = self.theta0 + self.lf_s*logistic_f(-self.RTheta*self.lf_I0)
        self.has_sig = float(self.lf_s != 0)
        self.invTauRiseTheta = self.invTauRiseTheta*self.has_sig
        if self.lf_s == 0.0:
            self.thetaInit = 0.0
            self.theta = 0.0
            self.normDeltaTheta = self.DeltaTheta / (self.thetaMax - self.theta0) # normalizing constant for theta increment; VT is the theta0
        #with open('parameter_test_neuron.txt','w') as f:
        #    print(self.__dict__,file=f)
        #with open('parameter_test_default.txt','w') as f:
        #    print(eifdtboundsigklrArgs,file=f)
        if setIC:
            self.SetIC(VInit=VInit,gInit=gInit,thetaInit=thetaInit,thetasInit=thetasInit)
        #print('theta0 = %.8f'%self.theta0)
        #print('thetaInit = %.8f'%self.thetaInit)
        #print('theta_s Init = %.8f'%self.thetasInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        #s = 0.0#float(self.gK > 0.0) #float(Iext==0) #
        self.dV1 = (self.Vb - self.V + self.has_sig*self.DeltaT * my_exp((self.V - self.VT)/self.DeltaT) + self.R*Iext - self.R*self.gK*(self.V-self.EK))*self.invTau
        #self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(self.V/self.R-self.lf_I0)) - self.theta*(1.0-s))*self.invTauRiseTheta - self.invTauTheta*s*self.theta
        self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.theta)*self.invTauRiseTheta # - self.invTauTheta*s*self.theta
        self.dths1 = (self.theta0*(1.0-self.has_sig) - self.thetas) * self.invTauTheta # this model reduces to LIFDTBoundKLR if the sigmoid amplitude is zero (has_sig == 0)
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        #s = 0.0#float(self.gK1 > 0.0) #float(Iext==0) #
        th1 = self.theta + self.dt * self.dth1
        self.dV2 = (self.Vb - self.V1 + self.has_sig*self.DeltaT * my_exp((self.V1 - self.VT)/self.DeltaT) + self.R*Iext - self.R*self.gK1*(self.V1-self.EK))*self.invTau
        #self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(self.V1/self.R-self.lf_I0)) - th1*(1.0-s))*self.invTauRiseTheta - self.invTauTheta*s*th1
        self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - th1)*self.invTauRiseTheta # - self.invTauTheta*s*th1
        self.dths2 = (self.theta0*(1.0-self.has_sig) - (self.thetas + self.dt * self.dths1) ) * self.invTauTheta # this model reduces to LIFDTBoundKLR if the sigmoid amplitude is zero (has_sig == 0)
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.thetas = self.thetas + (self.dths1 + self.dths2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > (self.theta + self.thetas):
            self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV
            self.thetas += self.normDeltaTheta * (self.thetaMax - self.thetas) # limits the growth of theta up to thetaMax
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

    def GetThreshold(self):
        return self.theta + self.thetas

    def CalcRealThreshold(self):
        if self.has_sig:
            return calc_real_exp_threshold(self.Vb,self.DeltaT,self.VT)
        else:
            return self.thetas
    
    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.thetas = self.thetasInit
        self.gK = self.gInit
        self.Vrec = self.V

    def GetRestingState(self,Iext=0.0):
        if self.has_sig:
            Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,self.VT,returnRest=True)
        else:
            Vs = super().GetRestingState(Iext)['V']
        return dict(     V=Vs,
                     theta=self.has_sig*(self.theta0+self.lf_s*logistic_f(self.RTheta*(Iext - self.lf_I0))),
                    thetas=(1.0-self.has_sig)*self.theta0,
                        gK=0.0)

    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta,thetasInit=self.thetas,gInit=self.gK)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### EIFHDTBoundSigKLRIA neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class EIFHDTBoundSigKLRIA(LIFDTBoundedKLR):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,DeltaT,VT,lf_s,lf_I0,Rth,tauRiseTheta,thetasInit,hIAInit,gIA,AmIA,kmIA,VmIA,AhIA,khIA,VhIA,tauhIA,EKIA,noiseSignal,setIC=True):
        LIFDTBoundedKLR.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,noiseSignal,setIC=False)
        self.lf_s = lf_s
        self.lf_I0 = lf_I0
        self.RTheta = Rth
        self.DeltaT = DeltaT
        self.VT = VT
        self.thetasInit = thetasInit
        self.thetas = thetasInit
        self.invTauRiseTheta = 1.0/tauRiseTheta
        self.dths1 = 0.0
        self.dths2 = 0.0
        self.normDeltaTheta = DeltaTheta / thetaMax # normalizing constant for theta increment
        self.thetaRest = self.theta0 + self.lf_s*logistic_f(-self.RTheta*self.lf_I0)
        self.hIAInit = hIAInit
        self.gIA = gIA
        self.AmIA = AmIA
        self.kmIA = kmIA
        self.VmIA = VmIA
        self.invTauhIA = 1.0/tauhIA
        self.AhIA = AhIA
        self.khIA = khIA
        self.VhIA = VhIA
        self.EKIA = EKIA
        self.hIA = hIAInit
        self.dh1 = 0.0
        self.dh2 = 0.0
        #with open('parameter_test_neuron.txt','w') as f:
        #    print(self.__dict__,file=f)
        #with open('parameter_test_default.txt','w') as f:
        #    print(eifdtboundsigklrArgs,file=f)
        if setIC:
            self.SetIC(VInit=VInit,hIAInit=hIAInit,gInit=gInit,thetaInit=thetaInit,thetasInit=thetasInit)
        #print('theta0 = %.8f'%self.theta0)
        #print('thetaInit = %.8f'%self.thetaInit)
        #print('theta_s Init = %.8f'%self.thetasInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        s = 0.0#float(self.gK > 0.0) #float(Iext==0) #
        minf = self.AmIA*boltzmann_f(self.kmIA*(self.V-self.VmIA))
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.VT)/self.DeltaT) + self.R*(Iext - self.gK*(self.V-self.EK) - self.gIA*minf*self.hIA*(self.V-self.EKIA) ))*self.invTau
        self.dh1 = (self.AhIA*boltzmann_f(self.khIA*(self.V-self.VhIA))-self.hIA)*self.invTauhIA
        #self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(self.V/self.R-self.lf_I0)) - self.theta*(1.0-s))*self.invTauRiseTheta - self.invTauTheta*s*self.theta
        self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.theta*(1.0-s))*self.invTauRiseTheta - self.invTauTheta*s*self.theta
        self.dths1 = - self.thetas * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        s = 0.0#float(self.gK1 > 0.0) #float(Iext==0) #
        th1 = self.theta + self.dt * self.dth1
        h1 = self.hIA + self.dt * self.dh1
        minf = self.AmIA*boltzmann_f(self.kmIA*(self.V1-self.VmIA))
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.VT)/self.DeltaT) + self.R*(Iext - self.gK1*(self.V1-self.EK) - self.gIA*minf*h1*(self.V1-self.EKIA) ))*self.invTau
        self.dh1 = (self.AhIA*boltzmann_f(self.khIA*(self.V1-self.VhIA))-h1)*self.invTauhIA
        #self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(self.V1/self.R-self.lf_I0)) - th1*(1.0-s))*self.invTauRiseTheta - self.invTauTheta*s*th1
        self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - th1*(1.0-s))*self.invTauRiseTheta - self.invTauTheta*s*th1
        self.dths2 = - (self.thetas + self.dt * self.dths1) * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.hIA = self.hIA + (self.dh1 + self.dh2) * self.dt / 2.0
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.thetas = self.thetas + (self.dths1 + self.dths2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > (self.theta + self.thetas):
            self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV
            self.thetas += self.normDeltaTheta * (self.thetaMax - self.thetas) # limits the growth of theta up to thetaMax
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

    def GetThreshold(self):
        return self.theta + self.thetas

    def CalcRealThreshold(self):
        hinf = lambda V: self.AhIA*boltzmann_f_np(self.khIA*(V-self.VhIA))
        hinfprime = lambda V: self.AhIA*self.khIA*boltzmann_fprime_np(self.khIA*(V-self.VhIA))
        minf = lambda V: self.AmIA*boltzmann_f_np(self.kmIA*(V-self.VmIA))
        minfprime = lambda V: self.AmIA*self.kmIA*boltzmann_fprime_np(self.kmIA*(V-self.VmIA))
        RIA = lambda V: self.R * self.gIA * minf(V) * hinf(V)
        RIAprime = lambda V: self.R * self.gIA * ( minfprime(V)*hinf(V) + hinfprime(V)*minf(V) )
        return calc_real_exp_threshold(self.Vb,self.DeltaT,self.VT,RIA=RIA,RIAprime=RIAprime,EKIA=self.EKIA)
    
    def Reset(self):
        self.V = self.VInit
        self.hIA = self.hIAInit
        self.theta = self.thetaInit
        self.thetas = self.thetasInit
        self.gK = self.gInit
        self.Vrec = self.V
    
    def GetG2(self):
        return self.hIA

    def GetRestingState(self,Iext=0.0):
        hinf = lambda V: self.AhIA*boltzmann_f_np(self.khIA*(V-self.VhIA))
        hinfprime = lambda V: self.AhIA*self.khIA*boltzmann_fprime_np(self.khIA*(V-self.VhIA))
        minf = lambda V: self.AmIA*boltzmann_f_np(self.kmIA*(V-self.VmIA))
        minfprime = lambda V: self.AmIA*self.kmIA*boltzmann_fprime_np(self.kmIA*(V-self.VmIA))
        RIA = lambda V: self.R * self.gIA * minf(V) * hinf(V)
        RIAprime = lambda V: self.R * self.gIA * ( minfprime(V)*hinf(V) + hinfprime(V)*minf(V) )
        Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,self.VT,returnRest=True,RIA=RIA,RIAprime=RIAprime,EKIA=self.EKIA)
        return dict(     V=Vs,
                       hIA=hinf(Vs),
                     theta=self.theta0+self.lf_s*logistic_f(self.RTheta*(Iext - self.lf_I0)),
                    thetas=0.0,
                        gK=0.0)

    def GetStateAsIC(self):
        return dict(VInit=self.V,hIAInit=self.hIA,thetaInit=self.theta,thetasInit=self.thetas,gInit=self.gK)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### AdEIFHDTBoundSigKLR neuron
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class AdEIFHDTBoundSigKLR(LIFDTBoundedKLR):
    def __init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,DeltaT,VT,lf_s,lf_I0,Rth,tauRiseTheta,thetasInit,wInit,gW,tauW,DeltaW,noiseSignal,setIC=True):
        LIFDTBoundedKLR.__init__(self,dt,VInit,thetaInit,gInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,thetaMax,EK,tauK,DeltaGK,rv,deltaV,noiseSignal,setIC=False)
        self.lf_s = lf_s
        self.lf_I0 = lf_I0
        self.RTheta = Rth
        self.DeltaT = DeltaT
        self.VT = VT
        self.thetasInit = thetasInit
        self.thetas = thetasInit
        self.invTauRiseTheta = 1.0/tauRiseTheta
        self.dths1 = 0.0
        self.dths2 = 0.0
        self.normDeltaTheta = DeltaTheta / thetaMax # normalizing constant for theta increment
        self.thetaRest = self.theta0 + self.lf_s*logistic_f(-self.RTheta*self.lf_I0)
        self.w = wInit
        self.wInit = wInit
        self.gW = gW
        self.invTauW = 1.0/tauW
        self.DeltaW = DeltaW
        self.dw1 = 0.0
        self.dw2 = 0.0
        self.has_sig = float(self.lf_s != 0)
        self.invTauRiseTheta = self.invTauRiseTheta*self.has_sig
        if self.lf_s == 0:
            self.thetaInit = 0.0
            self.theta = 0.0
            self.normDeltaTheta = self.DeltaTheta / (self.thetaMax - self.theta0) # normalizing constant for theta increment; VT is the theta0
        #with open('parameter_test_neuron.txt','w') as f:
        #    print(self.__dict__,file=f)
        #with open('parameter_test_default.txt','w') as f:
        #    print(eifdtboundsigklrArgs,file=f)
        if setIC:
            self.SetIC(VInit=VInit,wInit=wInit,gInit=gInit,thetaInit=thetaInit,thetasInit=thetasInit)
        #print('theta0 = %.8f'%self.theta0)
        #print('thetaInit = %.8f'%self.thetaInit)
        #print('theta_s Init = %.8f'%self.thetasInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        #s = 0.0#float(self.gK > 0.0) #float(Iext==0) #
        self.dV1 = (self.Vb - self.V + self.has_sig*self.DeltaT * my_exp((self.V - self.VT)/self.DeltaT) + self.R*(Iext - self.gK*(self.V-self.EK) - self.w))*self.invTau
        self.dw1 = (self.gW * (self.V - self.Vb) - self.w) * self.invTauW
        #self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(self.V/self.R-self.lf_I0)) - self.theta*(1.0-s))*self.invTauRiseTheta - self.invTauTheta*s*self.theta
        self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.theta)*self.invTauRiseTheta #- self.invTauTheta*s*self.theta
        self.dths1 = (self.theta0*(1.0-self.has_sig) - self.thetas) * self.invTauTheta # this model reduces to LIFDTBoundKLR if the sigmoid amplitude is zero (has_sig == 0)
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.gK1 = self.gK + self.dt * self.dg1
        #s = 0.0#float(self.gK1 > 0.0) #float(Iext==0) #
        th1 = self.theta + self.dt * self.dth1
        w1 = self.w + self.dt*self.dw1
        self.dV2 = (self.Vb - self.V1 + self.has_sig*self.DeltaT * my_exp((self.V1 - self.VT)/self.DeltaT) + self.R*(Iext - self.gK1*(self.V1-self.EK) - w1))*self.invTau
        #self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(self.V1/self.R-self.lf_I0)) - th1*(1.0-s))*self.invTauRiseTheta - self.invTauTheta*s*th1
        self.dw2 = (self.gW * (self.V1 - self.Vb) - w1) * self.invTauW
        self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - th1)*self.invTauRiseTheta #- self.invTauTheta*s*th1
        self.dths2 = (self.theta0*(1.0-self.has_sig) - (self.thetas + self.dt * self.dths1) ) * self.invTauTheta # this model reduces to LIFDTBoundKLR if the sigmoid amplitude is zero (has_sig == 0)
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.w = self.w + (self.dw1 + self.dw2) * self.dt / 2.0
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.thetas = self.thetas + (self.dths1 + self.dths2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0

    def Step(self,Iext = 0.0):
        if self.V > (self.theta + self.thetas):
            self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV
            self.thetas += self.normDeltaTheta * (self.thetaMax - self.thetas) # limits the growth of theta up to thetaMax
            self.gK += self.DeltaGK
            self.w += self.DeltaW
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

    def GetThreshold(self):
        return self.theta + self.thetas

    def CalcRealThreshold(self):
        if self.has_sig:
            return calc_real_exp_threshold(self.Vb,self.DeltaT,self.VT,Rgw=self.R*self.gW)
        else:
            return self.thetas
    
    def Reset(self):
        self.V = self.VInit
        self.w = self.wInit
        self.theta = self.thetaInit
        self.thetas = self.thetasInit
        self.gK = self.gInit
        self.Vrec = self.V
    
    def GetG2(self):
        return self.w

    def GetRestingState(self,Iext=0.0):
        if self.has_sig:
            Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,self.VT,returnRest=True,Rgw=self.R*self.gW)
        else:
            Vs = super().GetRestingState(Iext)['V']
        return dict(     V=Vs,
                         w=self.gW*(Vs - self.Vb),
                     theta=self.has_sig*(self.theta0+self.lf_s*logistic_f(self.RTheta*(Iext - self.lf_I0))),
                    thetas=(1.0-self.has_sig)*self.theta0,
                        gK=0.0)

    def GetStateAsIC(self):
        return dict(VInit=self.V,wInit=self.w,thetaInit=self.theta,thetasInit=self.thetas,gInit=self.gK)


"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Exponential LIF neuron with (natural) Dynamic Threshold
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class EIFDT(LIF):
    #   __init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC = True):
    def __init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,noiseSignal,theta0=None,Vpeak=None,setIC=True):
        if not (theta0 is None):
            if theta0 != VT:
                print(' ... WARNING ::: using theta0 for VT')
                VT = theta0
        LIF.__init__(self,dt,VInit,Vr,Vb,tau,R,thetaInit,noiseSignal,setIC=False)
        self.thetaInit = thetaInit
        self.DeltaT = DeltaT
        self.V1 = 0.0
        self.theta0 = VT
        self.invTauTheta = 1.0 / tauTheta
        self.DeltaTheta = DeltaTheta
        self.th1 = 0.0
        self.dth1 = 0.0
        self.dth2 = 0.0
        if (Vpeak is None) or numpy.isnan(Vpeak):
            Vs = self.CalcRealThreshold()
            Vpeak = Vs + 20.0 # in mV
            #print('exponential threshold = %f' % self.Vpeak)
        self.Vpeak = Vpeak
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.theta)/self.DeltaT) + self.R*Iext)*self.invTau
        self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.th1 = self.theta + self.dt*self.dth1
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.th1)/self.DeltaT) + self.R*Iext)*self.invTau
        self.dth2 = (self.theta0 - self.th1) * self.invTauTheta
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
    
    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.Vrec = self.V

    def Step(self,Iext = 0.0):
        if self.V > self.Vpeak:
            self.V = self.Vr
            self.theta += self.DeltaTheta
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def CalcRealThreshold(self):
        return calc_real_exp_threshold(self.Vb,self.DeltaT,self.theta)
    
    def GetRestingState(self,Iext=0.0):
        return dict(V=calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,self.theta0,returnRest=True),theta=self.theta0)
    
    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Exponential LIF neuron with (natural) Dynamic Threshold (subthreshold dynamic threshold)
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class EIFSubDT(EIFDT):
    #   __init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC = True):
    def __init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,G2,thetaInit,VT,tauTheta,DeltaTheta,noiseSignal,theta0=None,Vpeak=None,setIC=True):
        EIFDT.__init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,noiseSignal,theta0=theta0,Vpeak=Vpeak,setIC=False)
        self.G2 = G2
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.theta)/self.DeltaT) + self.R*Iext)*self.invTau
        # the threshold equation is 1 time step delayed as compared to the V equation
        # this avoids exploding threshold due to the term V - Vb
        # the divergence is specially apparent for 0 < DeltaT < 1
        # the strictly correct equation is the one commented out below
        #self.dth1 = (self.theta0 - self.theta + self.G2 * (self.V - self.Vb)) * self.invTauTheta
        self.dth1 = (self.theta0 - self.theta + self.G2 * (self.Vrec - self.Vb)) * self.invTauTheta # Vrec is the previous potential
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.th1 = self.theta + self.dt*self.dth1
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.th1)/self.DeltaT) + self.R*Iext)*self.invTau
        # the threshold equation is 1 time step delayed as compared to the V equation
        # this avoids exploding threshold due to the term V - Vb
        # the divergence is specially apparent for 0 < DeltaT < 1
        # the strictly correct equation is the one commented out below
        #self.dth2 = (self.theta0 - self.th1 + self.G2 * (self.V1 - self.Vb)) * self.invTauTheta
        self.dth2 = (self.theta0 - self.th1 + self.G2 * (self.V - self.Vb)) * self.invTauTheta
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
    
    def GetG2(self):
        return self.G2

    def Step(self,Iext = 0.0):
        if self.V > self.Vpeak:
            #print('V-Vb    = %.8e'%(self.V - self.Vb))
            #print('Vrec-Vb = %.8e'%(self.Vrec - self.Vb))
            self.V = self.Vr
            self.theta += self.DeltaTheta
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)
    
    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.Vrec = self.V

    def CalcRealThreshold(self):
        VT_rescaled = (self.G2 * self.Vb - self.theta0) / (1.0 - self.G2)
        DeltaT_rescaled = self.DeltaT / (1.0 - self.G2)
        return calc_real_exp_threshold(self.Vb,DeltaT_rescaled,VT_rescaled)
    
    def GetRestingState(self,Iext=0.0):
        VT_rescaled = (self.G2 * self.Vb - self.theta0) / (1.0 - self.G2)
        DeltaT_rescaled = self.DeltaT / (1.0 - self.G2)
        Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,DeltaT_rescaled,VT_rescaled,returnRest=True)
        theta_star = self.theta0 + self.G2 * (Vs - self.Vb)
        #print('    ....    V_s = %.2f        theta_s = %.2f'%(Vs,theta_star))
        return dict(V=Vs,theta=theta_star)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Exponential LIF neuron with bounded spike-dependent (natural) Dynamic Threshold + K-current + linear reset
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class EIFDTBoundKLR(EIFDT):
    #   __init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC = True):
    def __init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,DeltaGK,EK,tauK,gInit,deltaV,rv,thetaMax,noiseSignal,theta0=None,Vpeak=None,setIC=True):
        EIFDT.__init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,noiseSignal,theta0=theta0,Vpeak=Vpeak,setIC=False)

        # the K current parameters, IC and variables
        self.gInit = gInit
        self.EK = EK
        self.DeltaGK = DeltaGK
        self.invTauK = 1.0 / tauK
        self.gK = gInit
        self.dg1 = 0.0
        self.dg2 = 0.0
        self.gK1 = 0.0

        # the linear reset parameters
        self.rv = rv
        self.deltaV = deltaV

        # the bounded threshold parameters
        self.thetaMax = thetaMax
        self.normDeltaTheta = self.DeltaTheta / (self.thetaMax - self.theta0) # normalizing constant for theta increment; VT is the theta0

        #self.V1 = 0.0
        #self.th1 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit)

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.theta)/self.DeltaT) - self.R*self.gK*(self.V-self.EK) + self.R*Iext)*self.invTau
        self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.th1 = self.theta + self.dt*self.dth1
        self.gK1 = self.gK + self.dt * self.dg1
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.th1)/self.DeltaT) - self.R*self.gK1*(self.V1-self.EK) + self.R*Iext)*self.invTau
        self.dth2 = (self.theta0 - self.th1) * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0
        #print('(%.2f,%.2f,%.2f)' % (self.V,self.theta,self.gK))

    def Step(self,Iext = 0.0):
        if self.V > self.Vpeak:
            # the reset uses the previous potential, Vrec, to avoid divergence due to V - Vb
            # the divergence is specially apparent for 0 < DeltaT < 1
            # the strictly correct equation is the one commented out below
            #self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV # using the previous V (i.e., Vrec) because V maybe too high
            self.V = self.Vb + self.rv * (self.Vrec - self.Vb) - self.deltaV # using the previous V (i.e., Vrec) because V maybe too high
            self.theta += self.normDeltaTheta * (self.thetaMax - self.theta)
            #self.theta += self.DeltaTheta
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

    def GetGK(self):
        return self.gK

    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.gK = self.gInit
        self.Vrec = self.V

    def CalcRealThreshold(self):
        return calc_real_exp_threshold(self.Vb,self.DeltaT,self.theta0)
    
    def GetRestingState(self,Iext=0.0):
        Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,self.theta0,returnRest=True)
        return dict(V=Vs,theta=self.theta0,gK=0.0)
    
    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta,gInit=self.gK)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Exponential LIF neuron with bounded spike-dependent (natural) Dynamic Threshold + input-dependent threshold + K-current + linear reset
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class EIFDTBoundSigKLR(EIFDT):
    #   __init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC = True):
    def __init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,DeltaGK,EK,tauK,gInit,deltaV,rv,thetaMax,noiseSignal,theta0,lf_s,lf_I0,Rth,tauRiseTheta,thetasInit,Vpeak=None,setIC=True):
        EIFDTBoundKLR.__init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,DeltaGK,EK,tauK,gInit,deltaV,rv,thetaMax,noiseSignal,theta0=None,Vpeak=Vpeak,setIC=False)
        # input-dependent threshold
        self.theta0 = theta0
        self.lf_s = lf_s
        self.lf_I0 = lf_I0
        self.RTheta = Rth
        self.invTauRiseTheta = 1.0/tauRiseTheta
        #self.thetaRest = self.theta0 + self.lf_s*logistic_f(-self.RTheta*self.lf_I0)

        # spike-dependent threshold
        self.normDeltaTheta = DeltaTheta / thetaMax # normalizing constant for theta increment; here, thetas_0 = 0
        self.thetasInit = thetasInit
        self.thetas = thetasInit
        self.dths1 = 0.0
        self.dths2 = 0.0
        self.ths1 = 0.0

        # the total threshold
        self.VT = self.thetasInit + self.thetaInit

        #self.V1 = 0.0
        #self.th1 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit,thetasInit=thetasInit)

    def GetThreshold(self):
        return self.VT

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.VT)/self.DeltaT) - self.R*self.gK*(self.V-self.EK) + self.R*Iext)*self.invTau
        #self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.theta)*self.invTauRiseTheta
        #self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(-self.Vrec/self.R-self.lf_I0)) - self.theta)*self.invTauRiseTheta
        self.dths1 = -self.thetas * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.th1 = self.theta + self.dt*self.dth1
        self.gK1 = self.gK + self.dt * self.dg1
        self.ths1 = self.thetas + self.dt * self.dths1
        self.VT = self.th1 + self.ths1
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.VT)/self.DeltaT) - self.R*self.gK1*(self.V1-self.EK) + self.R*Iext)*self.invTau
        #self.dth2 = (self.theta0 - self.th1) * self.invTauTheta
        self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.th1)*self.invTauRiseTheta
        #self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(-self.V/self.R-self.lf_I0)) - self.th1)*self.invTauRiseTheta
        self.dths1 = -self.ths1 * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.thetas = self.thetas + (self.dths1 + self.dths2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0
        self.VT = self.theta + self.thetas
        #print('(%.2f,%.2f,%.2f)' % (self.V,self.theta,self.gK))

    def Step(self,Iext = 0.0):
        if self.V > self.Vpeak:
            # the reset uses the previous potential, Vrec, to avoid divergence due to V - Vb
            # the divergence is specially apparent for 0 < DeltaT < 1
            # the strictly correct equation is the one commented out below
            #self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV # using the previous V (i.e., Vrec) because V maybe too high
            self.V = self.Vb + self.rv * (self.Vrec - self.Vb) - self.deltaV # using the previous V (i.e., Vrec) because V maybe too high
            #print(self.normDeltaTheta * (self.thetaMax - self.thetas))
            self.thetas += self.normDeltaTheta * (self.thetaMax - self.thetas)
            #self.thetas += self.DeltaTheta
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

    def GetGK(self):
        return self.gK

    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.thetas = self.thetasInit
        self.gK = self.gInit
        self.Vrec = self.V

    def CalcRealThreshold(self):
        return calc_real_exp_threshold(self.Vb,self.DeltaT,self.VT)
    
    def GetRestingState(self,Iext=0.0):
        VT_rest = self.theta0+self.lf_s*logistic_f(self.RTheta*(Iext - self.lf_I0))
        Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,VT_rest,returnRest=True)
        #VT_f = lambda V: self.theta0+self.lf_s*logistic_f_np(-self.RTheta*(V/self.R + self.lf_I0))
        #VT_fprime = lambda V: -self.lf_s*logistic_fder_np(-self.RTheta*(V/self.R + self.lf_I0))*self.RTheta/self.R
        #Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,VT_f,returnRest=True,VTprime=VT_fprime)
        #VT_rest = VT_f(Vs)
        return dict(V=Vs,theta=VT_rest,thetas=0.0,gK=0.0)
    
    def SetIC(self,**kwargs):
        super().SetIC(**kwargs)
        self.VT = self.thetasInit + self.thetaInit

    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta,thetasInit=self.thetas,gInit=self.gK)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Exponential LIF neuron with bounded spike-dependent (natural) Dynamic Threshold + V-dependent threshold + K-current + linear reset
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class EIFDTBoundSigVKLR(EIFDT):
    #   __init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC = True):
    def __init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,DeltaGK,EK,tauK,gInit,deltaV,rv,thetaMax,noiseSignal,theta0,lf_s,lf_I0,Rth,tauRiseTheta,thetasInit,Vpeak=None,setIC=True):
        EIFDTBoundKLR.__init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,DeltaGK,EK,tauK,gInit,deltaV,rv,thetaMax,noiseSignal,theta0=None,Vpeak=Vpeak,setIC=False)
        # input-dependent threshold
        self.theta0 = theta0
        self.lf_s = lf_s
        self.lf_I0 = lf_I0
        self.RTheta = Rth
        self.invTauRiseTheta = 1.0/tauRiseTheta
        #self.thetaRest = self.theta0 + self.lf_s*logistic_f(-self.RTheta*self.lf_I0)

        # spike-dependent threshold
        self.normDeltaTheta = DeltaTheta / thetaMax # normalizing constant for theta increment; here, thetas_0 = 0
        self.thetasInit = thetasInit
        self.thetas = thetasInit
        self.dths1 = 0.0
        self.dths2 = 0.0
        self.ths1 = 0.0

        # the total threshold
        self.VT = self.thetasInit + self.thetaInit

        #self.V1 = 0.0
        #self.th1 = 0.0
        if setIC:
            self.SetIC(VInit=VInit,thetaInit=thetaInit,gInit=gInit,thetasInit=thetasInit)

    def GetThreshold(self):
        return self.VT

    def Integrate_RK2(self,Iext):
        Iext += self.SumInput()
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.VT)/self.DeltaT) - self.R*self.gK*(self.V-self.EK) + self.R*Iext)*self.invTau
        #self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        #self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.theta)*self.invTauRiseTheta
        self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(-self.Vrec/self.R-self.lf_I0)) - self.theta)*self.invTauRiseTheta
        self.dths1 = -self.thetas * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.th1 = self.theta + self.dt*self.dth1
        self.gK1 = self.gK + self.dt * self.dg1
        self.ths1 = self.thetas + self.dt * self.dths1
        self.VT = self.th1 + self.ths1
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.VT)/self.DeltaT) - self.R*self.gK1*(self.V1-self.EK) + self.R*Iext)*self.invTau
        #self.dth2 = (self.theta0 - self.th1) * self.invTauTheta
        #self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.th1)*self.invTauRiseTheta
        self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(-self.V/self.R-self.lf_I0)) - self.th1)*self.invTauRiseTheta
        self.dths1 = -self.ths1 * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.thetas = self.thetas + (self.dths1 + self.dths2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0
        self.VT = self.theta + self.thetas
        #print('(%.2f,%.2f,%.2f)' % (self.V,self.theta,self.gK))

    def Step(self,Iext = 0.0):
        if self.V > self.Vpeak:
            # the reset uses the previous potential, Vrec, to avoid divergence due to V - Vb
            # the divergence is specially apparent for 0 < DeltaT < 1
            # the strictly correct equation is the one commented out below
            #self.V = self.Vb + self.rv * (self.V - self.Vb) - self.deltaV # using the previous V (i.e., Vrec) because V maybe too high
            self.V = self.Vb + self.rv * (self.Vrec - self.Vb) - self.deltaV # using the previous V (i.e., Vrec) because V maybe too high
            #print(self.normDeltaTheta * (self.thetaMax - self.thetas))
            self.thetas += self.normDeltaTheta * (self.thetaMax - self.thetas)
            #self.thetas += self.DeltaTheta
            self.gK += self.DeltaGK
            self.Vrec = 60.0 # previous state of V was a spike with V=60mV
        else:
            self.Vrec = self.V
            self.Integrate_RK2(Iext)

    def GetGK(self):
        return self.gK

    def Reset(self):
        self.V = self.VInit
        self.theta = self.thetaInit
        self.thetas = self.thetasInit
        self.gK = self.gInit
        self.Vrec = self.V

    def CalcRealThreshold(self):
        return calc_real_exp_threshold(self.Vb,self.DeltaT,self.VT)
    
    def GetRestingState(self,Iext=0.0):
        #VT_rest = self.theta0+self.lf_s*logistic_f(self.RTheta*(Iext - self.lf_I0))
        #Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,VT_rest,returnRest=True,VTprime=fprime)
        VT_f = lambda V: self.theta0+self.lf_s*logistic_f_np(-self.RTheta*(V/self.R + self.lf_I0))
        VT_fprime = lambda V: -self.lf_s*logistic_fder_np(-self.RTheta*(V/self.R + self.lf_I0))*self.RTheta/self.R
        Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,VT_f,returnRest=True,VTprime=VT_fprime)
        VT_rest = VT_f(Vs)
        return dict(V=Vs,theta=VT_rest,thetas=0.0,gK=0.0)
    
    def SetIC(self,**kwargs):
        super().SetIC(**kwargs)
        self.VT = self.thetasInit + self.thetaInit

    def GetStateAsIC(self):
        return dict(VInit=self.V,thetaInit=self.theta,thetasInit=self.thetas,gInit=self.gK)

"""
############################################################################################################
############################################################################################################
############################################################################################################
###
###
### Exponential LIF neuron with bounded spike-dependent (natural) Dynamic Threshold + input-dependent threshold + K-current + linear reset + IA current
###
###
############################################################################################################
############################################################################################################
############################################################################################################
"""

class EIFDTBoundSigKLRIA(EIFDTBoundSigKLR):
    #   __init__(self,dt,VInit,thetaInit,Vr,Vb,tau,R,theta0,tauTheta,DeltaTheta,noiseSignal,setIC = True):
    def __init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,DeltaGK,EK,tauK,gInit,deltaV,rv,thetaMax,noiseSignal,theta0,lf_s,lf_I0,Rth,tauRiseTheta,thetasInit,hIAInit,gIA,AmIA,kmIA,VmIA,AhIA,khIA,VhIA,tauhIA,EKIA,Vpeak=None,setIC=True):
        EIFDTBoundSigKLR.__init__(self,dt,VInit,Vr,Vb,tau,R,DeltaT,thetaInit,VT,tauTheta,DeltaTheta,DeltaGK,EK,tauK,gInit,deltaV,rv,thetaMax,noiseSignal,theta0,lf_s,lf_I0,Rth,tauRiseTheta,thetasInit,Vpeak=Vpeak,setIC=False)
        self.hIAInit = hIAInit
        self.gIA = gIA
        self.AmIA = AmIA
        self.kmIA = kmIA
        self.VmIA = VmIA
        self.invTauhIA = 1.0/tauhIA
        self.AhIA = AhIA
        self.khIA = khIA
        self.VhIA = VhIA
        self.EKIA = EKIA
        self.hIA = hIAInit
        self.dh1 = 0.0
        self.dh2 = 0.0

        if setIC:
            self.SetIC(VInit=VInit,hIAInit=hIAInit,thetaInit=thetaInit,thetasInit=thetasInit,gInit=gInit)

    def Integrate_RK2(self,Iext):
        #print(self.V)
        Iext += self.SumInput()
        minf = self.AmIA*boltzmann_f(self.kmIA*(self.Vrec-self.VmIA))
        self.dV1 = (self.Vb - self.V + self.DeltaT * my_exp((self.V - self.VT)/self.DeltaT) + self.R*(Iext - self.gK*(self.V-self.EK) - self.gIA*minf*self.hIA*(self.V-self.EKIA)) )*self.invTau
        self.dh1 = (self.AhIA*boltzmann_f(self.khIA*(self.Vrec-self.VhIA))-self.hIA)*self.invTauhIA
        #self.dth1 = (self.theta0 - self.theta) * self.invTauTheta
        self.dth1 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.theta)*self.invTauRiseTheta
        self.dths1 = -self.thetas * self.invTauTheta
        self.dg1 = -self.gK * self.invTauK
        self.V1 = self.V + self.dt*self.dV1 + self.noiseSignal()
        self.th1 = self.theta + self.dt*self.dth1
        self.gK1 = self.gK + self.dt * self.dg1
        self.ths1 = self.thetas + self.dt * self.dths1
        self.VT = self.th1 + self.ths1
        h1 = self.hIA + self.dt * self.dh1
        minf = self.AmIA*boltzmann_f(self.kmIA*(self.V-self.VmIA))
        self.dV2 = (self.Vb - self.V1 + self.DeltaT * my_exp((self.V1 - self.VT)/self.DeltaT) + self.R*(Iext - self.gK1*(self.V1-self.EK) - self.gIA*minf*h1*(self.V1-self.EKIA)) )*self.invTau
        self.dh1 = (self.AhIA*boltzmann_f(self.khIA*(self.V-self.VhIA))-h1)*self.invTauhIA
        #self.dth2 = (self.theta0 - self.th1) * self.invTauTheta
        self.dth2 = (self.theta0 + self.lf_s*logistic_f(self.RTheta*(Iext-self.lf_I0)) - self.th1)*self.invTauRiseTheta
        self.dths1 = -self.ths1 * self.invTauTheta
        self.dg2 = -self.gK1 * self.invTauK
        self.V = self.V + (self.dV1 + self.dV2) * self.dt / 2.0 + self.noiseSignal()
        self.hIA = self.hIA + (self.dh1 + self.dh2) * self.dt / 2.0
        self.theta = self.theta + (self.dth1 + self.dth2) * self.dt / 2.0
        self.thetas = self.thetas + (self.dths1 + self.dths2) * self.dt / 2.0
        self.gK = self.gK + (self.dg1 + self.dg2) * self.dt / 2.0
        self.VT = self.theta + self.thetas
        #print('(%.2f,%.2f,%.2f)' % (self.V,self.theta,self.gK))

    def Reset(self):
        self.V = self.VInit
        self.hIA = self.hIAInit
        self.theta = self.thetaInit
        self.thetas = self.thetasInit
        self.gK = self.gInit
        self.Vrec = self.V

    def CalcRealThreshold(self):
        hinf = lambda V: self.AhIA*boltzmann_f_np(self.khIA*(V-self.VhIA))
        hinfprime = lambda V: self.AhIA*self.khIA*boltzmann_fprime_np(self.khIA*(V-self.VhIA))
        minf = lambda V: self.AmIA*boltzmann_f_np(self.kmIA*(V-self.VmIA))
        minfprime = lambda V: self.AmIA*self.kmIA*boltzmann_fprime_np(self.kmIA*(V-self.VmIA))
        RIA = lambda V: self.R * self.gIA * minf(V) * hinf(V)
        RIAprime = lambda V: self.R * self.gIA * ( minfprime(V)*hinf(V) + hinfprime(V)*minf(V) )
        return calc_real_exp_threshold(self.Vb,self.DeltaT,self.VT,RIA=RIA,RIAprime=RIAprime,EKIA=self.EKIA) #calc_real_exp_threshold(self.Vb,self.DeltaT,self.VT)
    
    def GetG2(self):
        return self.hIA

    def GetRestingState(self,Iext=0.0):
        VT_rest = self.theta0+self.lf_s*logistic_f(self.RTheta*(Iext - self.lf_I0))
        hinf = lambda V: self.AhIA*boltzmann_f_np(self.khIA*(V-self.VhIA))
        hinfprime = lambda V: self.AhIA*self.khIA*boltzmann_fprime_np(self.khIA*(V-self.VhIA))
        minf = lambda V: self.AmIA*boltzmann_f_np(self.kmIA*(V-self.VmIA))
        minfprime = lambda V: self.AmIA*self.kmIA*boltzmann_fprime_np(self.kmIA*(V-self.VmIA))
        RIA = lambda V: self.R * self.gIA * minf(V) * hinf(V)
        RIAprime = lambda V: self.R * self.gIA * ( minfprime(V)*hinf(V) + hinfprime(V)*minf(V) )
        Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,VT_rest,returnRest=True,RIA=RIA,RIAprime=RIAprime,EKIA=self.EKIA)
        #Vs = calc_real_exp_threshold(self.Vb+self.R*Iext,self.DeltaT,VT_rest,returnRest=True)
        return dict(V=Vs,hIA=hinf(Vs),theta=VT_rest,thetas=0.0,gK=0.0)

    def GetStateAsIC(self):
        return dict(VInit=self.V,hIAInit=self.hIA,thetaInit=self.theta,thetasInit=self.thetas,gInit=self.gK)