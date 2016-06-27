#import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

python3 = sys.version_info > (3,0)

def length(a):
    '''Alias for len'''
    return len(a)

def mat_var(data,struc,field):
    '''Get the 'field' from the 'struc' in the 'data' imported from matlab'''
    return data[struc][field].item()

def compress_seq(seq):
    '''Get changes seq[i]!=seq[i+1] in 'seq' as (place=i, new_value=seq[i+1])'''
    num = np.array(range(0,len(seq)))
    change = seq[1:]!=seq[:-1]
    change = np.hstack(([True],change))
    return np.column_stack((num[change],seq[change]))

def crop_series(seq, limits=(None,None), axis=None):
    '''Normalize the series into [limits:0,limits:1] via cropping'''
    (lmin, lmax) = limits
    ret = seq
    if lmin is not None:
        ret[seq<lmin] = lmin
    if lmax is not None:
        ret[seq>lmax] = lmax
    return ret

def adjust_series(seq, axis=None):
    '''Normalize the series into [0,1] via a shift and a multiplication'''
    if axis is None:
        smin = np.nanmin(seq)
        smax = np.nanmax(seq)
    else:
        smin = np.nanmin(seq, axis, keepdims=True)
        #smin = np.expand_dims(smin, axis)
        smax = np.nanmax(seq, axis, keepdims=True)
        #smax = np.expand_dims(smax, axis)
    ret = (seq-smin)*(1.0/(smax-smin))
    return ret

def shrink_series(seq, axis=None):
    '''Normalize the series into [0,1] via a shift and a multiplication'''
    if axis is None:
        smax = np.nanmax(seq)
    else:
        smax = np.nanmax(seq, axis, keepdims=True)
        #smax = np.expand_dims(smax, axis)
    ret = seq*(1.0/smax)
    return ret
    
def normalize_series(seq, axis=None):
    '''Normalize the series E=0, D2=1 via a shift and a multiplication'''
    if axis is None:
        smean = np.nanmean(seq)
        sdev = np.nanstd(seq) 
    else:
        smean = np.nanmean(seq, axis, keepdims=True)
        #smean = np.expand_dims(smean, axis)
        sdev = np.nanstd(seq, axis, keepdims=True)
        #sdev = np.expand_dims(sdev, axis)
    ret = (seq-smean)*(1.0/(sdev))
    return ret

def square_angle(x,y):
    '''return square-shaped angle between (0,8)'''
    #reg = (3 if y<0 else 2) if x<0 else (4 if y<0 else 1)
    reg = np.ones((len(x),))
    reg[x*y<0] += 1
    reg[y<0] += 2
    sub = np.sign(x)*y-np.sign(y)*x
    return reg*2 + sub - 1
    
def mx_plot(iterable, shape=None, cbar=True):
    '''Plot multiple matrices'''
    if shape is None:
        shape = [1, len(iterable)]
    fg = plt.figure(figsize=(shape[1]*4, shape[0]*4))
    c = 1
    for i in iterable:
        ax = plt.subplot(shape[0], shape[1], c)
        ca = ax.matshow(i)
        cbar and fg.colorbar(ca)
        c = c+1
        
def bool_to_ord(arr):
    '''Boolean selector from ordinal numbers'''
    seq = np.arange(0,len(arr))
    idx = np.array(arr, dtype=bool)
    return seq[idx]

def ord_to_bool(seq, maximum = None):
    '''Get ordinal numbers from a boolean array'''
    if maximum is None:
        maximum = max(seq)
    arr = np.zeros((maximum+1,))
    arr[seq] = 1
    return arr

def unfold(seq, nanvalue = None):
    '''Unfold data represented on a [0, 2pi) torus, caalculating the number of spins'''
    reg = np.array(seq)
    reg[reg is nanvalue] = np.nan
    reg = np.array(pd.Series(reg).fillna(method='pad').values)
    diff = np.hstack(([0], reg[1:]-reg[:-1]))
    jump = np.abs(diff) > np.pi
    cw = np.sign(diff)
    cw[~jump] = 0
    spin = np.cumsum(-cw*2*np.pi)
    return seq + spin


def transition_graph(seq):
    '''Transition graph of a chain with frequencies'''
    df = pd.DataFrame()
    df['from'] = np.array(seq[:-1])
    df['to'] = np.array(seq[1:])
    df['count'] = 1
    return df.groupby(['from', 'to']).sum().reset_index()

def mat_to_df(data, ref_column_name = None, exclude_fields = [], collect_mishaped = False):
    '''Convert a matlab structured array to a DataFrame'''
    df = pd.DataFrame()
    not_fit = {}
    length = None if ref_column_name is None else len(data[ref_column_name].item())
    for key in data.dtype.names :
        if (key not in exclude_fields) :
            valid = (type(data[key].item()) is np.ndarray)
            if (length is None and valid):
                length = len(data[key].item())
            else:
                valid = valid and (len(data[key].item()) == length)

            if (valid) :
                df[key] = data[key].item()
            else:
                not_fit[key] = data[key].item()

    if collect_mishaped:
        return df, not_fit
    else:
        return df

def mat_files(directory, pattern = '.mat$'):
    ''' get files matching pattern'''
    import re
    from os import walk, path
    subdir = []
    name = []
    full = []
    directory = path.realpath(directory)
    for (dirpath, dirnames, filenames) in walk(directory):
        files = list(filter(re.compile(pattern).search, filenames))
        full.extend([path.join(dirpath, f) for f in files])
        name.extend(files)
        subdir.extend([dirpath.replace(directory,'') for f in files])
    return full, subdir, name

def strictly_increasing(L):
    '''test series'''
    return all(x<y for x, y in zip(L[:-1], L[1:]))

def strictly_decreasing(L):
    '''test series'''
    return all(x>y for x, y in zip(L[:-1], L[1:]))

def non_increasing(L):
    '''test series'''
    return all(x>=y for x, y in zip(L[:-1], L[1:]))

def non_decreasing(L):
    '''test series'''
    return all(x<=y for x, y in zip(L[:-1], L[1:]))

def hls_matrix(h,l,s):
    '''map hls to rgb'''
    from colorsys import hls_to_rgb
    
    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    
    return c

def colorize(z):
    '''nice colors for complx numbers'''
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8
    
    return hls_matrix(h,l,s)
