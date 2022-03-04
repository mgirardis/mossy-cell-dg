import math
import numpy
#from scipy.io import savemat
import scipy.stats
import scipy.signal
#import neuronpy.graphics as spikeplot
import neo
from modules.misc import get_empty_list
#import elephant.spike_train_correlation as spkc
#import elephant.conversion as elec
import quantities

def get_n_max_corrcoef(C,n=10,i=None,j=None):
    if type(C) is list:
        i,j = numpy.nonzero(numpy.tril(numpy.ones(C[0].shape)))
        ncf = get_empty_list(len(C))
        ind = get_empty_list(len(C))
        linind = get_empty_list(len(C))
        for k,c in enumerate(C):
            ncf[k],ind[k],linind[k] = get_n_max_corrcoef(c,n,i=i,j=j)
    else:
        CC = C.copy()
        if (i is None) or (j is None):
            i,j = numpy.nonzero(numpy.tril(numpy.ones(C[0].shape)))
        CC[i,j] = -numpy.inf
        CC[numpy.isnan(CC)] = -numpy.inf
        CC = CC.flatten()
        k = numpy.argsort(CC)[-n:]
        ncf = CC[k]
        linind = k
        z = numpy.unravel_index(k,C.shape)
        k = numpy.lexsort((z[1],z[0]))
        ncf = ncf[k]
        linind = linind[k]
        ind = [ (z[0][i],z[1][i]) for i in k ]
        #ind = [ (i,j) for i,j in zip(z[0],z[1]) ]
    return ncf,ind,linind

def calc_correlation_distribution(C,nbins=100,smooth=False,ignoreZeroCorr=True):
    if smooth is True:
        smooth = 'average'
    if type(C) is list:
        n = len(C)
        P = get_empty_list(n)
        bins = get_empty_list(n)
        avg = get_empty_list(n)
        std = get_empty_list(n)
        for i,c in enumerate(C):
            P[i],bins[i],avg[i],std[i] = calc_correlation_distribution(c,nbins=nbins,smooth=smooth,ignoreZeroCorr=ignoreZeroCorr)
        return P, bins, avg, std
    else:
        x = C[numpy.eye(C.shape[0])!=1]
        if ignoreZeroCorr:
            x = x[x!=0]
        P,bins = numpy.histogram(x,bins=nbins,density=True)
        if smooth == 'savgol':
            P = scipy.signal.savgol_filter(P, 5, 2)
        elif smooth == 'average':
            P = moving_average(P,n=10)
        P = P / numpy.sum(P)
        avg = numpy.nanmean(x)
        std = numpy.nanstd(x)
        return P, bins[:-1], avg, std

def calc_null_correlation(S,ntrials=None,smooth=False,smoothSigma=None,binarize=True,overlap=False,spk_threshold=59.0,rowvar=False,nandiag=True):
    if ntrials is None:
        ntrials = S.shape[0]
    A,_ = calc_correlation_matrices(S,smooth=smooth,smoothSigma=smoothSigma,binarize=binarize,overlap=overlap,spk_threshold=spk_threshold,rowvar=rowvar,nandiag=nandiag)
    C = A.copy()
    i,j = numpy.nonzero(numpy.triu(numpy.ones(A.shape),k=1))
    for k in range(ntrials):
        C += rand_corr_matrix(A.copy(),i=i,j=j)
    return C / ntrials

def calc_correlation_matrices(S,DeltaT=None,smooth=False,smoothSigma=0.1,binarize=True,overlap=False,spk_threshold=59.0,rowvar=False,nandiag=True,filterSpkFreq=False):
    # DeltaT is an interval length in terms of the rows indices of S
    # smooth = False or 'mexican' or 'gaussian' for smoothing spikes
    # if rowvar == true -> time series are rows instead of columns
    if rowvar:
        S = numpy.transpose(S)
    if DeltaT is None:
        DeltaT = S.shape[0]
    if binarize:
        S = get_binary_spike_series(S,spk_threshold=spk_threshold)
    if filterSpkFreq:
        S = filter_spk_freq(S,filterSpkFreq)
    if smooth:
        S = smooth_spikes(S,smoothFunc=smooth,stddev=smoothSigma)
    if overlap:
        nT = S.shape[0] - 1
        get_interval = get_time_interval_overlap
    else:
        nT = int(numpy.ceil(float(S.shape[0]) / float(DeltaT)))
        get_interval = get_time_interval_adjacent
    
    C = get_empty_list(nT)
    tRange = get_empty_list(nT)
    for i in range(nT):
        t1,t2 = get_interval(i,DeltaT)
        if t2 > S.shape[0]:
            t2 = S.shape[0]
        if (t2-t1) > 2:
            tRange[i] = (t1,t2)
            C[i] = numpy.corrcoef(S[t1:t2,:],rowvar=rowvar)
            C[i][numpy.isnan(C[i])] = 0.0
            #if numpy.count_nonzero(numpy.isnan(C[i])) > 0:
            #print('index == %d -> [%d;%d]     ---- number of NaN: %d' % (i,t1,t2,numpy.count_nonzero(numpy.isnan(C[i]))))
            if nandiag:
                numpy.fill_diagonal(C[i],numpy.nan)

    if nT == 1:
        C = C[0]
        tRange = tRange[0]
    else:
        C = [c for c in C if c is not None]
        tRange = [t for t in tRange if t is not None] 
    return C, tRange

def filter_spk_freq(S,f):
    """ to be implemented: remove background spikes using median filter """
    n = S.shape[1]
    for i in range(n):
        S[:,i] = scipy.signal.medfilt(S[:,i],3)
    return S

def filter_null_avg(C,null_avg):
    if type(C) is list:
        for i in range(len(C)):
            C[i] = filter_null_avg(C[i],null_avg)
    else:
        C[numpy.nonzero(C < null_avg)] = 0.0
    return C

def get_time_interval_overlap(i,DeltaT):
    return i, i+DeltaT

def get_time_interval_adjacent(i,DeltaT):
    return i*DeltaT, (i+1)*DeltaT

def rand_corr_matrix(A,i=None,j=None):
    # keeps diagonal in place
    # randomize only upper triangular part, since A is symmetrical
    # i and j are row and col indices of upper triangular part of A
    if i is None or j is None:
        i,j = numpy.nonzero(numpy.triu(numpy.ones(A.shape),k=1))
    x = A[i,j] # gets all the upper triangular elements in A
    x = x[numpy.random.permutation(len(x))]
    A[i,j] = x
    return numpy.triu(A,k=1) + numpy.tril(numpy.transpose(A))

def get_binary_spike_series(data,spk_threshold=59.0):
    return numpy.asarray(data>spk_threshold,dtype=float)

def smooth_spikes(data,smoothFunc=None,dt=0.01,stddev=0.1):
    # considering each column in data is a binary (1,0) spike time series
    # convolves each column of data with smoothFunc (if not provided, assumed to be Gaussian)
    # if Gaussian is assumed, time scale = 0.01, and std dev = 0.1
    N = data.shape[1]
    if smoothFunc is None:
        smoothFunc = get_gaussian_kernel(stddev,dt=dt)
    else:
        if smoothFunc == 'gaussian':
            smoothFunc = get_gaussian_kernel(stddev,dt=dt)
        elif smoothFunc == 'mexican':
            smoothFunc = get_mexican_hat_kernel(stddev, dt=dt)
    for i in range(N):
        data[:,i] = numpy.convolve(data[:,i],smoothFunc,mode='same')
    return data

def moving_average(x, n=10):
    return numpy.convolve(x,numpy.ones(n)/n,mode='same')

def get_mexican_hat_kernel(sigma1, J=None, dt=None):
    if J is None:
        J=4*sigma1
    if dt is None:
        dt = 0.001
    sigma2 = numpy.sqrt(numpy.power(sigma1, 2.) + numpy.power(J, 2.))
    if sigma2 < sigma1:
        temp = sigma2.copy()
        sigma2 = sigma1.copy()
        sigma1 = temp
    k1 = get_gaussian_kernel(sigma1,dt=dt)
    k2 = get_gaussian_kernel(sigma2,dt=dt)
    n2 = k2.shape[0]
    n1 = k1.shape[0]
    m = int( numpy.floor((n2-n1)/2.0) )
    n = int( numpy.ceil((n2-n1)/2.0) )
    return numpy.pad(k1,(m,n))-k2

def get_gaussian_kernel(sigma,dt=0.01):
    t = numpy.arange(-3*sigma,3*sigma,dt)
    G = scipy.stats.norm.pdf(t,scale=sigma) * dt
    return G / numpy.sum(G)

def membpotential_to_spiketrain(data,t=None,spk_threshold=59.0):
    # converts each column of data into a Neo.SpikeTrain
    # t is the time vector
    (T,N) = data.shape
    spktrains = get_empty_list(N)
    if t is None:
        t = numpy.arange(T)
    #dt = numpy.mean(numpy.squeeze(numpy.diff(t)))
    for j in range(N):
        spktrains[j] = neo.SpikeTrain(t[numpy.nonzero(data[:,j] > spk_threshold)], units='ms', t_start=t[0], t_stop=t[-1])
    return spktrains

def membpotential_to_spike_times(data,t=None,spk_threshold=59.0):
    # converts each column of data into an array of spike times
    # t is the time vector
    (T,N) = data.shape
    spktimes = get_empty_list(N)
    if t is None:
        t = numpy.arange(T)
    for j in range(N):
        spktimes[j] = t[numpy.nonzero(data[:,j] > spk_threshold)]
    return spktimes

def get_unique_pair_indices(k):
    if not (type(k) is list):
        _,a = numpy.unique(k, axis=0, return_inverse=True)
        return k
    n = len(k)
    a = numpy.array(k).flatten()
    _,a = numpy.unique(a, axis=0, return_inverse=True)
    return split_array(a,n)

def split_array(a, n):
    k, m = divmod(len(a), n)
    return numpy.array([ a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n) ])

def flatten_tuple_list(l):
    r = []
    for e in l:
        if type(e) is list:
            r += flatten_tuple_list(e)
        else:
            r.append(e)
    return r

def sort_tuple_list(l,k):
    return [ l[i] for i in k ]
