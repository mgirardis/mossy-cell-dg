import os
import scipy.io

def create_fig_data_dir(file_prefix,dir_suffix,neuronType,paramset,saveFig,saveData):
    if paramset == 'none':
        file_prefix += '_' + neuronType
    else:
        file_prefix += '_' + paramset
    outFileNamePrefix_fig,outFileNamePrefix_data = file_prefix,file_prefix
    fig_dir = 'fig_' + dir_suffix
    data_dir = 'data_' + dir_suffix
    if saveFig:
        outFileNamePrefix_fig = os.path.join(fig_dir,file_prefix)
        if not os.path.isdir(fig_dir):
            print('*** creating dir: %s' % fig_dir)
            os.mkdir(fig_dir)
    if saveData:
        outFileNamePrefix_data = os.path.join(data_dir,file_prefix)
        if not os.path.isdir(data_dir):
            print('*** creating dir: %s' % data_dir)
            os.mkdir(data_dir)
    return outFileNamePrefix_fig,outFileNamePrefix_data

def save_fig(plt,file_prefix,file_suffix='',resolution=72,file_format='png',verbose=True):
    if plt is None:
        import matplotlib.pyplot as plt
    fn1 = check_and_get_filename(file_prefix + file_suffix + '.' + file_format)
    if verbose:
        print('*** saving %s' % fn1)
    plt.savefig(fname=fn1, dpi=resolution, format=file_format)
    return fn1

def save_data(data,file_name_or_prefix,file_suffix='',file_format='npz',numpy=None,append=False,verbose=True):
    # file format may be 'mat' as well
    # data is a dict of the form {'varName':varValue}
    if numpy is None:
        import numpy
    if file_format.lower() == 'npz':
        fn1 = check_and_get_filename(file_name_or_prefix + file_suffix + '.' + file_format)
        if verbose:
            print('*** saving %s' % fn1)
        if append:
            raise ValueError('not possible to append to numpy npz file')
        numpy.savez(fn1,**data)
    elif file_format.lower() == 'mat':
        if append:
            if verbose:
                print('*** appending to %s' % file_name_or_prefix)
            with open(file_name_or_prefix,'ab') as f:
                scipy.io.savemat(f, data)   # appe
        else:
            fn1 = check_and_get_filename(file_name_or_prefix + file_suffix + '.' + file_format)
            if verbose:
                print('*** saving %s' % fn1)
            scipy.io.savemat(fn1,data)
    else:
        raise ValueError('file_format should be npz or mat')


def pair_to_str(fmt,p):
    if (type(p) is list) and (type(p[0]) is tuple):
        s = [None] * len(p)
        for i,pp in enumerate(p):
            s[i] = pair_to_str(fmt,pp)
        return s
    return fmt % (p[0],p[1])

def check_and_get_filename(fname):
    if os.path.isfile(fname):
        fn = os.path.splitext(fname)
        k = 0
        while True:
            k += 1
            f = fn[0] + ('_%d' % k) + fn[1]
            if not os.path.isfile(f):
                break
        return f
    return fname
