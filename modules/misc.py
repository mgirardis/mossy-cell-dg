import copy
import modules.input as inp
import numpy
import scipy.signal
import collections

def calc_spike_features(t,I,V,th,I0_values,simArgs,tauTheta=None,Vth=59.0,nPoints_maxcurve=15,thresholdmethod='model',**findpeaks_args):
    # thresholdmethod -> 'model', 'maxcurvature', 'minderivative'
    thresholdmethod = thresholdmethod.lower()
    if not ( thresholdmethod in ['model', 'maxcurvature', 'minderivative']):
        raise ValueError('thresholdmethod must be either model, maxcurvature, or minderivative')
    if not numpy.isscalar(I0_values):
        # returns matrices for each avg, std and spk, instead of vectors
        if type(I0_values) is list:
            I0_values = numpy.asarray(I0_values).flatten()
        n = len(I0_values)
        if (len(V) != n) or (len(th) != n):
            raise ValueError('the V or th list has to match the I list')
        outputDict = {'avg': get_empty_list(n), 'std': get_empty_list(n), 'spk': get_empty_list(n), 'I': I0_values.copy()}
        ISIstruct = copy.deepcopy(outputDict)
        AHPMinStruct = copy.deepcopy(outputDict)
        SpkThreshStruct = copy.deepcopy(outputDict)
        SpkThreshDiffStruct = copy.deepcopy(outputDict)
        AHPAmpStruct = copy.deepcopy(outputDict)
        DeltaThetaStruct = copy.deepcopy(outputDict)
        keys = [ k for k in outputDict.keys() if k!='I' ]
        for i in range(n):
            isi_temp,ahpmin_temp,spkth_temp,ahpamp_temp,dth_temp,spkthdiff_temp = calc_spike_features(t,I[i],V[i],th[i],I0_values[i],simArgs,tauTheta=tauTheta,Vth=Vth,nPoints_maxcurve=nPoints_maxcurve,thresholdmethod=thresholdmethod,**findpeaks_args)
            for k in keys:
                ISIstruct[k][i] = copy.deepcopy(isi_temp[k])
                AHPMinStruct[k][i] = copy.deepcopy(ahpmin_temp[k])
                SpkThreshStruct[k][i] = copy.deepcopy(spkth_temp[k])
                SpkThreshDiffStruct[k][i] = copy.deepcopy(spkthdiff_temp[k])
                AHPAmpStruct[k][i] = copy.deepcopy(ahpamp_temp[k])
                DeltaThetaStruct[k][i] = copy.deepcopy(dth_temp[k])
        for k in keys: # makes each column of each k matrix correspond to a current I[i]
            ISIstruct[k] = asarray_nanfill(ISIstruct[k]).T
            AHPMinStruct[k] = asarray_nanfill(AHPMinStruct[k]).T
            SpkThreshStruct[k] = asarray_nanfill(SpkThreshStruct[k]).T
            SpkThreshDiffStruct[k] = asarray_nanfill(SpkThreshDiffStruct[k]).T
            AHPAmpStruct[k] = asarray_nanfill(AHPAmpStruct[k]).T
            DeltaThetaStruct[k] = asarray_nanfill(DeltaThetaStruct[k]).T
        # the spk # is the mode across columns of each spk# matrix
        ISIstruct['spk'] = scipy.stats.mode(ISIstruct['spk'],axis=1,nan_policy='omit').mode.data.flatten() if ISIstruct['spk'].size > 0 else numpy.asarray([])
        AHPMinStruct['spk'] = scipy.stats.mode(AHPMinStruct['spk'],axis=1,nan_policy='omit').mode.data.flatten() if ISIstruct['spk'].size > 0 else numpy.asarray([])
        SpkThreshStruct['spk'] = scipy.stats.mode(SpkThreshStruct['spk'],axis=1,nan_policy='omit').mode.data.flatten() if SpkThreshStruct['spk'].size > 0 else numpy.asarray([])
        SpkThreshDiffStruct['spk'] = scipy.stats.mode(SpkThreshDiffStruct['spk'],axis=1,nan_policy='omit').mode.data.flatten() if SpkThreshDiffStruct['spk'].size > 0 else numpy.asarray([])
        AHPAmpStruct['spk'] = scipy.stats.mode(AHPAmpStruct['spk'],axis=1,nan_policy='omit').mode.data.flatten() if AHPAmpStruct['spk'].size > 0 else numpy.asarray([])
        DeltaThetaStruct['spk'] = scipy.stats.mode(DeltaThetaStruct['spk'],axis=1,nan_policy='omit').mode.data.flatten() if DeltaThetaStruct['spk'].size > 0 else numpy.asarray([])
    else:
        tsStart,tsEnd = find_first_last_idx(I>0)

        outputDict = {'avg': None, 'std': None, 'spk': None, 'I': I0_values}
        ISIstruct = copy.deepcopy(outputDict)
        AHPMinStruct = copy.deepcopy(outputDict)
        SpkThreshStruct = copy.deepcopy(outputDict)
        SpkThreshDiffStruct = copy.deepcopy(outputDict)
        AHPAmpStruct = copy.deepcopy(outputDict)
        DeltaThetaStruct = copy.deepcopy(outputDict)

        use_findpeaks = thresholdmethod != 'model'
        findpeaks_args = inp.set_default_kwargs(findpeaks_args,prominence=Vth)

        ISI = calc_variation_at_spk(t, V, Vth=Vth, tsStart=tsStart, tsEnd=tsEnd, use_findpeaks=use_findpeaks, findpeaks_args=findpeaks_args)
        ISIstruct['avg'] = numpy.nanmean(asarray_nanfill(ISI),axis=0)
        ISIstruct['std'] = numpy.nanstd(asarray_nanfill(ISI),axis=0)
        ISIstruct['spk'] = numpy.arange(len(ISIstruct['avg']))+1

        ahp_min = calc_min_between_spikes(V,V,Vth=Vth,tsStart=tsStart,tsEnd=tsEnd, use_findpeaks=use_findpeaks, findpeaks_args=findpeaks_args)
        AHPMinStruct['avg'] = numpy.nanmean(asarray_nanfill(ahp_min),axis=0)
        AHPMinStruct['std'] = numpy.nanstd(asarray_nanfill(ahp_min),axis=0)
        AHPMinStruct['spk'] = numpy.arange(len(AHPMinStruct['avg']))

        if thresholdmethod == 'model':
            spk_th = get_values_at_spike(th,V,Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,tsDelay=-1, use_findpeaks=use_findpeaks, findpeaks_args=findpeaks_args)
        else:
            if thresholdmethod == 'maxcurvature':
                spk_th = calc_threshold_max_curvature(V,squeeze_len1=False,nPoints=nPoints_maxcurve,use_findpeaks=True,V0=Vth,**findpeaks_args)
            else:
                spk_th = calc_threshold_min_derivative(V,squeeze_len1=False,nPoints=nPoints_maxcurve,use_findpeaks=True,V0=Vth,**findpeaks_args)
        spk_th_matrix = asarray_nanfill(spk_th)
        SpkThreshStruct['avg'] = numpy.nanmean(spk_th_matrix,axis=0)
        SpkThreshStruct['std'] = numpy.nanstd(spk_th_matrix,axis=0)
        SpkThreshStruct['spk'] = numpy.arange(len(SpkThreshStruct['avg']))

        SpkThreshDiffStruct['avg'] = numpy.nanmean(spk_th_matrix - spk_th_matrix[:,0].reshape((spk_th_matrix.shape[0],1)),axis=0) if spk_th_matrix.size > 0 else numpy.asarray([])
        SpkThreshDiffStruct['std'] = SpkThreshStruct['std']
        SpkThreshDiffStruct['spk'] = SpkThreshStruct['spk']

        ahp_amp = spk_th_matrix[:,:-1] - asarray_nanfill(ahp_min) if spk_th_matrix.size > 0 else spk_th_matrix
        AHPAmpStruct['avg'] = numpy.nanmean(ahp_amp,axis=0)
        AHPAmpStruct['std'] = numpy.nanstd(ahp_amp,axis=0)
        AHPAmpStruct['spk'] = numpy.arange(len(AHPAmpStruct['avg']))

        tauTheta = simArgs['tauTheta'] if tauTheta is None else tauTheta
        if thresholdmethod == 'model':
            dth_spk = [ (calc_DeltaTheta_experiments(spk_th[i],spk_th[i][0],tauTheta,ISI[i])[2] if spk_th[i].size > 0 else numpy.asarray([])) for i in range(len(spk_th))]
        else:
            dth_spk = [ (calc_DeltaTheta_experiments(spk_th[i][1:],spk_th[i][0],tauTheta,ISI[i],skip_del_first_element=True)[2] if spk_th[i].size > 0 else numpy.asarray([])) for i in range(len(spk_th))]
        DeltaThetaStruct['avg'] = numpy.nanmean(asarray_nanfill(dth_spk),axis=0)
        DeltaThetaStruct['std'] = numpy.nanstd(asarray_nanfill(dth_spk),axis=0)
        DeltaThetaStruct['spk'] = numpy.arange(len(DeltaThetaStruct['avg']))

    return ISIstruct,AHPMinStruct,SpkThreshStruct,AHPAmpStruct,DeltaThetaStruct,SpkThreshDiffStruct

def fit_tau_theta(DeltaT,th_amp,th_std,simArgs):
    fitFunc = lambda xx,a,b,c: a * numpy.exp(-b * xx) + c
    fitParam_model, _ = scipy.optimize.curve_fit(fitFunc, DeltaT, th_amp, p0=(1.0,1.0/simArgs['tauTheta'],simArgs['theta0']), maxfev=100000)
    return 1.0 / fitParam_model[1]

def calc_threshold_crossings(x,x_th,only_downward_crossing=False,only_upward_crossing=False):
    """
     returns all the indices of x when x crosses x_th
     if only_downward_crossing == True, then returns only the crossings in which x is decreasing
     if only_upward_crossing == True, then returns only the crossings in which x is increasing
    """
    f = (x[:-1]-x_th)*(x[1:]-x_th) # f<0 -> crossing of threshold;
    if only_downward_crossing:
        tind_crossing = numpy.nonzero(numpy.logical_and( f <= 0 , x[1:]<=x_th))[0] # index of the slowing down instant (down crossing), since f<=0 and v[n] < v_th < v[n-1]
    else:
        if only_upward_crossing:
            tind_crossing = numpy.nonzero(numpy.logical_and( f <= 0 , x[:-1]<=x_th))[0] # index of the slowing up instant (down crossing), since f<=0 and v[n-1] < v_th < v[n]
        else:
            #t_cross,_ = find_inter_func(track.time,track.velocity,v_th) # finds all crossings of velocity and v_th using Newton's method
            tind_crossing = numpy.nonzero(f <= 0)[0] # index of all crossings
    return tind_crossing

def calc_threshold_min_derivative(V,V_peak_fraction=0.033,squeeze_len1_list_input=True,squeeze_len1=True,nPoints=15,return_time_idx=False,use_findpeaks=True,V0=None,**findpeaks_args):
    # method by Azouz & Gray 2000 PNAS, copied from Anh-Tuan implementation in MATLAB
    # if no findpeaks, then threshold data at V0 and use only instants in which V crosses V0 while increasing
    if type(V) is list:
        n = len(V)
        theta = get_empty_list(n)
        tspk = get_empty_list(n)
        for i in range(n):
            theta[i],tspk[i] = calc_threshold_min_derivative(V[i],V_peak_fraction=V_peak_fraction,squeeze_len1=squeeze_len1_list_input,nPoints=nPoints,return_time_idx=True,use_findpeaks=use_findpeaks,V0=V0,**findpeaks_args)
    else:
        if use_findpeaks:
            k,_ = scipy.signal.find_peaks(V,**findpeaks_args)
        else:
            if V0:
                k = calc_threshold_crossings(V,V0,only_downward_crossing=False,only_upward_crossing=True)
            else:
                k = [len(V)]
                nPoints = k[0] - 1
        n = len(k)
        theta = numpy.zeros(n)
        tspk = numpy.zeros(n,dtype=int)
        for i in range(n):
            # method of the maximum curvature, pg 46 Vinicius Lima MSc dissertation:
            # get the V that maximizes the function Kp = (d2V/dt2) * ( 1 + (dV/dt)**2 )**(-3/2)
            dV_th = V[k[i]] * V_peak_fraction
            start_idx = k[i] - nPoints
            start_idx = start_idx if start_idx >= 0 else 0
            end_idx = k[i] if k[i] < len(V) else len(V)-1
            Vspike = V[start_idx:end_idx].copy()
            dV = numpy.diff(Vspike,1) # dV/dt
            m = numpy.argmin(numpy.abs(dV-dV_th))
            theta[i] = Vspike[m].copy()
            tspk[i] = numpy.max((k[i] - nPoints,0)) + m
            #if tspk[i] == 0:
            #    print('oi')
        if len(theta) == 1 and squeeze_len1:
            theta = theta[0]
            tspk = tspk[0]
    if return_time_idx:
        return theta,tspk
    else:
        return theta


def calc_threshold_max_curvature(V,squeeze_len1_list_input=True,squeeze_len1=True,nPoints=15,return_time_idx=False,use_findpeaks=True,V0=None,**findpeaks_args):
    if type(V) is list:
        n = len(V)
        theta = get_empty_list(n)
        tspk = get_empty_list(n)
        for i in range(n):
            theta[i],tspk[i] = calc_threshold_max_curvature(V[i],squeeze_len1=squeeze_len1_list_input,nPoints=nPoints,return_time_idx=True,use_findpeaks=use_findpeaks,V0=V0,**findpeaks_args)
    else:
        if use_findpeaks:
            k,_ = scipy.signal.find_peaks(V,**findpeaks_args)
        else:
            if V0:
                k = calc_threshold_crossings(V,V0,only_downward_crossing=False,only_upward_crossing=True)
            else:
                k = [len(V)]
                nPoints = k[0] - 1
        n = len(k)
        theta = numpy.zeros(n)
        tspk = numpy.zeros(n)
        for i in range(n):
            # method of the maximum curvature, pg 46 Vinicius Lima MSc dissertation:
            # get the V that maximizes the function Kp = (d2V/dt2) * ( 1 + (dV/dt)**2 )**(-3/2)
            start_idx = k[i] - nPoints
            start_idx = start_idx if start_idx >= 0 else 0
            end_idx = k[i] if k[i] < len(V) else len(V)-1
            Vspike = V[start_idx:end_idx].copy()
            dV = numpy.diff(Vspike,1) # dV/dt
            d2V = numpy.diff(Vspike,2) # d2V/dt2
            Kp = d2V * (1.0 + dV[:-1]**2)**(-1.5)
            m = numpy.argmax(Kp)+1 if len(Kp) > 0 else 0
            theta[i] = Vspike[m].copy()
            tspk[i] = k[i] - nPoints + m
            #if tspk[i] == 0:
            #    print('oi')
        if len(theta) == 1 and squeeze_len1:
            theta = theta[0]
            tspk = tspk[0]
    if return_time_idx:
        return theta,tspk
    else:
        return theta

def remove_spikes(V,Vth=None,use_find_peaks=False,**findpeaks_args):
    if type(V) is list:
        n = len(V)
        VV = get_empty_list(n)
        for i in range(n):
            VV[i] = remove_spikes(V[i],Vth=Vth,use_find_peaks=use_find_peaks,**findpeaks_args)
        return VV
    else:
        if use_find_peaks:
            k_nan = numpy.isnan(V)
            if numpy.any(k_nan):
                V[k_nan] = -numpy.inf
            if not (Vth is None):
                findpeaks_args['threshold'] = Vth
            k,_ = scipy.signal.find_peaks(V,**findpeaks_args)
            k_logic = numpy.ones(len(V)) == 1
            k_logic[k] = False # remove V in the indices corresponding to the peaks in V
            return V[k_logic]
        else:
            Vth = 59.0 if Vth is None else Vth
            return V[numpy.nonzero(V<Vth)]

def find_first_last_idx(cond):
    cond = cond.flatten()
    kStart = numpy.argmax(cond)
    kEnd = len(cond) - numpy.argmax(cond[::-1]) - 1
    return kStart,kEnd

def get_values_at_spike(x,V,Vth=59.0,tsStart=None,tsEnd=None,tsDelay=0,stacklevel=0,use_findpeaks=False,findpeaks_args=None,noSpkValue=None):
    """
    returns values of x[t] for all t in which V[t]>Vth between t=tsStart and t=tsEnd
    if x is a list, do this for each time series in x
    if tsDelay is set, then returns x[t+tsDelay] instead of x[t]
    for instance, for value of x just before spike, use tsDelay=-1
    for value just after spike, use tsDelay=1
    """
    if type(x) is list:
        n = len(x)
        th_spk = get_empty_list(n)
        for i in range(n):
            if type(V) is list:
                th_spk[i] = get_values_at_spike(x[i],V[i],Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,tsDelay=tsDelay,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args,noSpkValue=noSpkValue)
            else:
                th_spk[i] = get_values_at_spike(x[i],V,Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,tsDelay=tsDelay,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args,noSpkValue=noSpkValue)
        return th_spk
    else:
        if type(V) is list and stacklevel == 0:
            return get_values_at_spike([x for i in range(len(V))],V,Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,tsDelay=tsDelay,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args,noSpkValue=noSpkValue)
        tsStart = 0 if tsStart is None else tsStart
        tsEnd = (len(V)-1) if tsEnd is None else tsEnd
        if type(V) is list:
            raise ValueError('V must be an ndarray the same shape as x')
        if use_findpeaks:
            tSpk,_ = scipy.signal.find_peaks(V,**findpeaks_args)
        else:
            tSpk = numpy.nonzero(V>Vth)[0]
        if (tSpk.size > 0) or (noSpkValue is None):
            tSpk = tSpk[ numpy.logical_and(tSpk >= tsStart, tSpk <= tsEnd) ] + tsDelay
            tSpk = tSpk[ numpy.logical_and(tSpk>=0, tSpk < len(V)) ]
            return x[tSpk]
        else:
            return numpy.asarray([noSpkValue]) if numpy.isscalar(noSpkValue) else numpy.asarray(noSpkValue)

def calc_variation_at_spk(x,V,Vth=59.0,tsStart=None,tsEnd=None,dx=1.0,stacklevel=0,use_findpeaks=False,findpeaks_args=None,noSpkValue=None):
    if type(x) is list:
        n = len(V)
        ISI = get_empty_list(n)
        for i in range(n):
            if type(V) is list:
                ISI[i] = calc_variation_at_spk(x[i],V[i],Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,dx=dx,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args,noSpkValue=noSpkValue)
            else:
                ISI[i] = calc_variation_at_spk(x[i],V,Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,dx=dx,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args,noSpkValue=noSpkValue)
        return ISI
    else:
        if type(V) is list and stacklevel == 0:
            return calc_variation_at_spk([x for i in range(len(V))],V,Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,dx=dx,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args,noSpkValue=noSpkValue)
        tsStart = 0 if tsStart is None else tsStart
        tsEnd = (len(V)-1) if tsEnd is None else tsEnd
        if type(V) is list:
            raise ValueError('V must be an ndarray the same shape as x')
        x_spk = get_values_at_spike(x,V,Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,tsDelay=0,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args,noSpkValue=noSpkValue)
        if (x_spk.size > 1) or (noSpkValue is None):
            return numpy.diff(x_spk).flatten()
        else:
            return (numpy.asarray([noSpkValue]) if numpy.isscalar(noSpkValue) else numpy.asarray(noSpkValue)) if x_spk.size == 0 else x_spk
        #return numpy.diff(x_spk).squeeze()

def calc_min_between_spikes(x,V,Vth=59.0,tsStart=None,tsEnd=None,stacklevel=0,use_findpeaks=False,findpeaks_args=None):
    """ minimize x[t] between every consecutive spikes in V[t] """
    if type(x) is list:
        n = len(x)
        min_spk = get_empty_list(n)
        for i in range(n):
            if type(V) is list:
                min_spk[i] = calc_min_between_spikes(x[i],V[i],Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args)
            else:
                min_spk[i] = calc_min_between_spikes(x[i],V,Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args)
        return min_spk
    else:
        if type(V) is list and stacklevel == 0:
            return calc_min_between_spikes([x for i in range(len(V))], V, Vth=Vth,tsStart=tsStart,tsEnd=tsEnd,stacklevel=stacklevel+1,use_findpeaks=use_findpeaks,findpeaks_args=findpeaks_args)
        tsStart = 0 if tsStart is None else tsStart
        tsEnd = (len(V)-1) if tsEnd is None else tsEnd
        if type(V) is list:
            raise ValueError('V must be an ndarray the same shape as x')
        if use_findpeaks:
            tSpk,_ = scipy.signal.find_peaks(V,**findpeaks_args)
        else:
            tSpk = numpy.nonzero(V>Vth)[0]
        tSpk = tSpk[ numpy.logical_and(tSpk >= tsStart, tSpk <= tsEnd) ]
        if len(tSpk) < 2:
            if len(tSpk) == 1:
                tSpk = numpy.asarray([0,tSpk[0]])
            else:
                tSpk = numpy.asarray([0,(len(V)-1)])
        return numpy.asarray([ numpy.min(x[t1:(t2+1)]) for t1,t2 in zip(tSpk[:-1],tSpk[1:]) ])

def calc_min_between_instants(x,tsStart=None,tsEnd=None):
    """minimize x[t] between t=tsStart and t=tsEnd; if x is list, do this for every element of x"""
    if type(x) is list:
        n = len(x)
        if tsStart is None:
            tsStart = get_empty_list(n)
        if tsEnd is None:
            tsEnd = get_empty_list(n)
        if numpy.isscalar(tsStart):
            tsStart = numpy.ones(n) * tsStart
        if numpy.isscalar(tsEnd):
            tsEnd = numpy.ones(n) * tsEnd
        m = get_empty_list(n)
        t = get_empty_list(n)
        for i in range(n):
            m[i],t[i] = calc_min_between_instants(x[i],tsStart=tsStart[i],tsEnd=tsEnd[i])
        return m,t
    else:
        tsStart = 0 if tsStart is None else tsStart
        tsEnd = (len(x)-1) if tsEnd is None else tsEnd
        t = numpy.nanargmin(x[tsStart:tsEnd])
        return x[t],t

def calc_DeltaTheta(th_at_reset,theta0,tauTheta,ISI):
    if not (type(ISI) is numpy.ndarray):
        ISI = numpy.asarray(ISI)
    if not (type(th_at_reset) is numpy.ndarray):
        th_at_reset = numpy.asarray(th_at_reset)
    m = len(ISI)
    if len(th_at_reset) <= m:
        raise ValueError('th_at_spk must have at least 1 entry more than ISI')
    # I use ISI[n-1] because python indexing is zero-based (starts at 0)
    # the same happens for ISI[:n] -> it returns all ISI up to ISI[n-1]
    ISIn = lambda i,n: ISI[i:n] if n > i else 0.0
    F = lambda n: 1.0 + numpy.sum([ numpy.exp(-numpy.sum(ISIn(i,n))/tauTheta) for i in range(n) ])
    dTheta = []
    for n in numpy.arange(m):
        dth = (th_at_reset[n] - theta0) / F(n) # using th_at_spk[n] instead of n+1 because python index starts at 0
        dTheta.append(dth)
    return numpy.nanmean(dTheta),numpy.nanstd(dTheta),dTheta

def calc_DeltaTheta_experiments(th_before_spk,theta0,tauTheta,ISI,skip_del_first_element=False):
    # this algorithm has to start with threshold of spike #1 (the second spike in the series)
    # corresponding to ISI[0] (between spike #0 and spike #1)
    if not (type(ISI) is numpy.ndarray):
        ISI = numpy.asarray(ISI)
    if not (type(th_before_spk) is numpy.ndarray):
        th_before_spk = numpy.asarray(th_before_spk)
    # I use ISI[n-1] because python indexing is zero-based (starts at 0)
    # the same happens for ISI[:n] -> it returns all ISI up to ISI[n-1]
    ISIn = lambda i,n: ISI[i:n] if n > i else 0.0
    F = lambda n: 1.0 + numpy.sum([ numpy.exp(-numpy.sum(ISIn(i,n))/tauTheta) for i in range(n) ])
    m = len(ISI)
    if (not skip_del_first_element) and (th_before_spk[0] == theta0):
        th_before_spk = th_before_spk[1:]
    dTheta = []
    for n in numpy.arange(m):
        dth = (th_before_spk[n] - theta0)*numpy.exp(ISI[n]/tauTheta) / F(n)
        dTheta.append(dth)
    return numpy.nanmean(dTheta),numpy.nanstd(dTheta),dTheta


def calc_theta_before_spk(th_at_spk,theta0,tauTheta,ISI):
    return numpy.asarray([ theta0+(th_at_spk[i]-theta0)*numpy.exp(-ISI[i]/tauTheta) for i in numpy.arange(len(th_at_spk[1:])) ])

def calc_noise_distribution(V,nbins_or_edges=10):
    if type(V) is list:
        n = len(V)
        if type(nbins_or_edges) is int:
            VV = asarray_nanfill(V).flatten()
            Vmin = numpy.nanmin(VV)
            Vmax = numpy.nanmax(VV)
            nbins = nbins_or_edges
            bins = numpy.linspace(Vmin,Vmax,nbins+1)
        else:
            nbins = len(nbins_or_edges) - 1
            bins = nbins_or_edges
        P = numpy.zeros(nbins)
        P2 = numpy.zeros(nbins)
        Vm = 0.0
        sd = 0.0
        for i in range(n):
            PP,_,VVm,ssd,_,_ = calc_noise_distribution(V[i],nbins_or_edges=bins)
            P += PP
            P2 += PP**2.0
            Vm += VVm
            sd += ssd
        P /= float(n)
        Vm /= float(n)
        sd /= float(n)
        Pmax = numpy.max(P)
        bins = bins[:-1]
        Pstd = numpy.sqrt(P2 / float(n) - P**2)
        return P,bins,Vm,sd,Pmax,Pstd
    else:
        Vm = numpy.mean(V)
        sd = numpy.std(V)
        P,bins = numpy.histogram(V,bins=nbins_or_edges,density=True)
        bins = bins[:-1]
        Pmax = numpy.max(P)
        return P,bins,Vm,sd,Pmax,P

def asarray_nanfill(x,return_empty_if_all_x_empty=False):
    return asarray_fill(x,v=numpy.nan,return_empty_if_all_x_empty=return_empty_if_all_x_empty)

def asarray_fill(x,v=None,return_empty_if_all_x_empty=False):
    """ x is a list of lists or ndarray's
        each element in x is converted to a row in the resulting ndarray
        each row in the resulting ndarray is completed with nan entries in order to match the number of elements
        the x list that has more elements """
    v = numpy.nan if v is None else v
    if type(x) is numpy.ndarray:
        return x
    if type(x) is list:
        if (type(x[0]) is int) or (type(x[0]) is float):
            return numpy.asarray(x)
    if not (type(x) is list):
        raise ValueError('x must be a list or an ndarray')
    get_len = lambda xx: 1 if numpy.isscalar(xx) else len(xx)
    N = numpy.max([ get_len(xx) for xx in x])
    if not return_empty_if_all_x_empty:
        if N == 0:
            N = 1
    return numpy.asarray([ fill_to_complete(xx,v,N) for xx in x ])

def fill_to_complete(x,v,N):
    """fills x with values v until len(x) == N"""
    if numpy.isscalar(x) or (x is None):
        x = copy.copy(x)
        x = [x] + [v for _ in range(N-1)]
        #x = repeat_to_complete([x],N)
    if type(x) is list:
        if N == len(x):
            return x
        y = get_empty_list(N,v)
        y[:len(x)] = x.copy()
        return y
    else:
        if N == len(x):
            return numpy.asarray(x).flatten()
        return numpy.concatenate((numpy.asarray(x).flatten(),numpy.full(N-len(x),v))).flatten()

def get_empty_list(n,v=None):
    return [v for i in range(n)]

def get_number_factors(n):
    i = int(n**0.5 + 0.5)
    while n % i != 0:
        i -= 1
    return int(i), int(n/i)

def extend_vec(x,N,fill_value=numpy.nan):
    """returns a vector of shape (N,), with its first elements given by x"""
    if numpy.isscalar(x):
        y = numpy.zeros(N) + fill_value
        y[0] = x
        return y
    if type(x) is list:
        y = [ fill_value for _ in range(N) ]
    if type(x) is numpy.ndarray:
        y = numpy.zeros(N) + fill_value
    y[:len(x)] = x
    return y

def repeat_to_complete(x,N,copydata=False):
# y = [ x(:)', x(:)', ..., x(:)' ]; such that y is [1,N] vector
    getX = lambda r: r
    if copydata:
        getX = lambda r: copy.deepcopy(r)
    if not(type(x) is list) and not(type(x) is numpy.ndarray):
        x = [getX(x)]
    if len(x) == 0:
        return x
    n = len(x)
    m = int(numpy.floor(N/n))
    if m < 1:
        return getX(x[:N])
    if type(x) is list:
        y = get_empty_list(m*n+N%n)
    elif type(x) is numpy.ndarray:
        y = numpy.zeros(m*n+N%n)
    for i in range(m):
        y[i*n:(i*n+n)] = getX(x)
    y[(m*n):] = getX(x[:(N-m*n)])
    return y

def expand_vec_into_mat(x,n,axis=0):
    """
    x is a vector or a list
    this function repeats x along axis n times to form a matrix with shape
    (n,len(x)) if axis==0
    (len(x),n) if axis==1
    """
    axis = int(axis)
    if axis < 0 or axis > 1:
        raise ValueError('axis can only be 0 for rows or 1 for cols')
    M = repeat_to_complete(x,len(x)*n).reshape((n,len(x)))
    if axis == 1:
        return M.T
    return M

def fix_dict_keys(di,k_old,k_new):
    """
    substitutes the keys in di dict given by k_old by the keys given by k_new
    di[k_old[i]] becomes di[k_new[i]]
    """
    if not(type(k_old) is list):
        k_old = [k_old]
    if not(type(k_new) is list):
        k_new = [k_new]
    if len(k_old) != len(k_new):
        raise ValueError('k_old and k_new must have the same number of elements')
    for i,k in enumerate(k_old):
        if k in di:
            di[k_new[i]] = di[k]
            di.pop(k,None)
    return di

def remove_key(di,k):
    di.pop(k,None)
    return di

def linearTransf(X,yLim=None,xLim=None,returnCoeff=False,coeff=None):
    """
    converts the values in X to the range of yLim by y = a + b*X
    if coeff is provided, then a,b=coeff
    otherwise, calculate the linear coefficients by xLim and yLim
    """
    if (yLim is None) and (coeff is None):
        return X
    if not (type(X) is numpy.ndarray):
        X = numpy.asarray(X)
    if coeff is None:
        if not yLim:
            return X
        xLim = [numpy.nanmin(X),numpy.nanmax(X)] if xLim is None else xLim
        b = numpy.asscalar(numpy.diff(yLim) / numpy.diff(xLim))
        a = yLim[0] - b * xLim[0]
    else:
        a,b = coeff
        if ((a is None) or (b is None)) and (not yLim):
            raise ValueError('if some coeff is None, then at least yLim must be provided')
        xLim = [numpy.nanmin(X),numpy.nanmax(X)] if xLim is None else xLim
        if b is None:
            b = numpy.asscalar(numpy.diff(yLim) / numpy.diff(xLim))
        if a is None:
            a = yLim[0] - b * xLim[0]
    if returnCoeff:
        return (a + b*X),(a,b)
    else:
        return a + b*X

def is_valid_input_range(rangeStr):
    nColon = len(find_all_char(rangeStr,':'))
    if nColon == 0:
        return False
    if nColon > 2:
        return False
    try:
        r = [ float(k) for k in rangeStr.split(':')  ]
    except:
        return False
    return True

def get_range_by_values(v,r=None):
    if not r:
        return 0,len(v)-1
    if type(r) is str:
        if ':' in r:
            vMin,vMax = r.split(':')
        else:
            raise ValueError('invalid range of values')
    else:
        if (type(r) is list) or (type(r) is tuple) or (type(r) is numpy.ndarray):
            vMin,vMax = r
        else:
            try:
                vMin,vMax = r
            except:
                vMin = 0
                vMax = float(r)
    vMin = float(vMin) if vMin else None
    vMax = float(vMax) if vMax else None
    if vMin:
        f = numpy.nonzero(v.flatten() == vMin)[0]
        if f.size > 0:
            i0 = f[0]
        else:
            i0 = 0
    else:
        i0 = 0
    if vMax:
        f = numpy.nonzero(v.flatten() == vMax)[0]
        if f.size > 0:
            i1 = f[0]
        else:
            i1 = len(v)-1
    else:
        i1 = len(v)-1
    return i0, i1
    

def get_range_from_string(rangeStr='0:1:11'):
    """
    format:
    1) a number -> returns the number
    2) start:end -> returns [start end]
    3) start:end:n -> returns a linspace from start to end, inclusive
    """
    nColon = len(find_all_char(rangeStr,':'))
    r = [ float(k) for k in rangeStr.split(':')  ]
    if nColon == 0:
        return float(rangeStr)
    elif nColon == 1:
        return numpy.linspace(r[0],r[1],2)
    else:
        return numpy.linspace(r[0],r[1],int(r[2]))

def is_valid_range_str(rangeStr):
    try:
        a = get_range_from_string(rangeStr)
        return True
    except:
        return False

def find_all_char(s,c):
    return [pos for pos, char in enumerate(s) if char == c]

def is_list_of_dict(d,internal_comparison_func=any):
    return (type(d) is list) and internal_comparison_func(type(r) is dict for r in d)

def is_list_of_list(l,internal_comparison_func=any):
    return (type(l) is list) and internal_comparison_func(type(r) is list for r in l)

def is_dict_of_list(d,internal_comparison_func=all):
    try:
        L = len(next(iter(d.values())))
    except:
        L = -1
    def safe_do(fun,*args):
        try:
            return fun(*args)
        except:
            return None
    return (type(d) is dict) and internal_comparison_func((type(d[k]) is list) and (safe_do(len,d[k]) == L) for k in d.keys())

def list_to_dict(l,key_prefix='key',start_at_one=True):
    """
    each item in l is converted into a key in the returning dict
    each key is labeled by key_prefix followed by their index in the list l
    l is either a list or a numpy.ndarray
    """
    return dict( zip( [ '{:s}{:d}'.format(key_prefix,k+1 if start_at_one else k) for k in range(len(l)) ], l ) )

def is_vector(v,fromnumpy=True):
    try:
        isSingleton = numpy.sum(numpy.asarray(numpy.asarray(v.shape) != 1,dtype=int)) == 1 # there only one dimension that has more than 1 elements
        isShapeLenOne = len(v.shape) == 1
    except AttributeError:
        if (not fromnumpy) and ((type(v) is list) or (type(v) is tuple)):
            return not numpy.isscalar(v)
        isShapeLenOne = False
        isSingleton = False
    return (not numpy.isscalar(v)) and (isShapeLenOne or isSingleton)

def get_hash_str(*args):
    h = 0
    for a in args:
        h += hash(str(a))
    return str(h)

def exists(v):
    return not(type(v) is type(None))

def dict_of_list_to_list_of_dict(v):
    """
    if all keys in v are lists of the same len,
    then converts v into a list o dict
    """
    if not is_dict_of_list(v):
        return v
    n = len(next(iter(v.values())))
    return [ {k:v[k][i] for k in v.keys()} for i in range(n) ]

class structtype(collections.abc.MutableMapping):
    def __init__(self,**kwargs):
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self,field,value):
        self.__dict__[field] = value
    def GetFields(self):
        return '; '.join([ k for k in self.__dict__.keys() if (k[0:2] != '__') and (k[-2:] != '__') ])
    def keys(self):
        return self.__dict__.keys()
    def items(self):
        return self.__dict__.items()
    def values(self):
        return self.__dict__.values()
    def pop(self,key,default_value=None):
        self.__dict__.pop(key,default_value)
    def __setitem__(self,label,value):
        self.__dict__[label] = value
    def __getitem__(self,label):
        return self.__dict__[label]
    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))
    def _get_kwargs(self):
        return sorted(self.__dict__.items())
    def _get_args(self):
        return []
    def __delitem__(self,*args):
        self.__dict__.__delitem__(*args)
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return iter(self.__dict__)

