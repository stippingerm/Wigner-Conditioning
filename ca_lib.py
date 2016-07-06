# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:47:42 2016

@author: Stippinger Marcell
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


### General access tools

class Bunch:
    '''Dot-access container'''
    def __init__(self, **kw):
        for name in kw:
            setattr(self, name, kw[name])
    def __repr__(self):
        return self.__dict__.__repr__()
    def __str__(self):
        return self.__dict__.__str__()


def test_hdf(filename):
    '''Test if HDF file exists and is available for reading'''
    try:
        if os.path.isfile(filename):
            with pd.HDFStore(filename, mode='r'):
                pass
            return True
        else:
            if os.path.isdir(filename):
                raise FileExistsError('Path beongs to directory.')
            else:
                return False
    except:
        try:
            os.remove(filename)
            warnings.warn('Corrupted file found, deleted.')
        except:
            raise FileExistsError('File is already open.')
        return False


### Constants for Losonczi Lab protocol

# Set event lengths
phases = ['Ready', 'CS', 'Trace', 'US', 'End']
phase_lookup={'Ready':0,'CS':1,'Trace':2,'US':3,'End':4}
durations=np.array([0,10,20,15,5])
events=np.cumsum(durations).astype(int)

legal_conditions=pd.MultiIndex.from_tuples([('Baseline','W+','A-'),
                              ('CS+','W+','A+'),('CS+','W+','A-'),
                              ('CS+','W-','A+'),('CS+','W-','A-'),
                              ('CS-','W+','A-'),('CS-','W-','A-')],
                              names=['context','port','puffed'])
legal_colors = ['y', 'magenta','purple','red','maroon','cyan','lime']

short_conditions=pd.MultiIndex.from_tuples([('Baseline','W+'),
                              ('CS+','W+'), ('CS+','W-'),
                              ('CS-','W+'), ('CS-','W-')],
                              names=['context','port'])
short_colors = ['y', 'magenta','red','cyan','lime']

# Set date formate
dtformat = '%Y-%m-%d-%Hh%Mm%Ss'
sessionbreak = np.timedelta64(8,'h')

epochs = pd.CategoricalIndex(['Pre-Learning', 'Learning', 'Post-Learning'], ordered=True)
contexts = pd.CategoricalIndex(['Baseline', 'CS+', 'CS-'], ordered=True)
port = pd.CategoricalIndex(['W+', 'W-'], ordered=True)
puff = pd.CategoricalIndex(['A+', 'A-'], ordered=True)

display_learning = ['learning_epoch','context','puffed','port']
sort_learning = ['learning_epoch','context','port','puffed']
sort_context = ['context','learning_epoch','port','puffed']


### Input handling for Losonczi Lab protocol

def __format_experiment_traits(df):
    '''Translate traits to human readable notation'''
    from datetime import datetime as dt
    et = df.rename(columns={'licking':'port','time':'timestr'})
    et['context'] = et['context'].replace('baseline','Baseline')
    et['port'] = et['port'].apply(lambda x: 'W+' if x else 'W-')
    et['puffed'] = et['puffed'].apply(lambda x: 'A+' if x else 'A-')
    et['session_num'] = et['session_num'].astype(int)
    et['datetime'] = et['timestr'].apply(lambda t: dt.strptime(t, dtformat))
    leapaday = (et['datetime'].values[1:]-et['datetime'].values[:-1]) > sessionbreak
    et['day_leap'] = np.append([True],leapaday)
    et['day_num'] = np.cumsum(np.append([0],leapaday.astype(int)))
    return et


def __load_fluor(mydir):
    '''Load the undocumented dict called fluor'''
    import pickle, sys
    pkl_file = open(os.path.join(mydir,'frame_fluor.pkl'), 'rb')

    if sys.version_info >= (3,5):
        # Python 3.5
        u = pickle._Unpickler(pkl_file)
        u.encoding = 'latin1'
        ret = u.load()
    else:
        # Python 2.7
        ret = pickle.load(pkl_file)

    return ret


def __create_mask(df, index=None, columns=None, threshold=0.9):
    '''Provide summary on data availability'''
    # How many ROIs are present in the given camera frame of a trial
    time_mask = df.reset_index().groupby(['time']).count()
    # Set True where at least 90% are present
    # NOTE: we mean 90% of ROIs that were not entirely silent in the trial
    time_mask = time_mask.drop(['roi_id'],axis=1).subtract(threshold*time_mask['roi_id'],axis=0)<0
    # In how many trials is the ROI present in the given camera frame
    roi_mask = df.reset_index().groupby(['roi_id']).count()
    # Set True those that are present in at least 90%
    # NOTE: we mean 90% of trials where the ROI was not entirely silent
    roi_mask = roi_mask.drop(['time'],axis=1).subtract(threshold*roi_mask['time'],axis=0)<0
    # Set all numeric entries to 0 (keep NaNs)
    time_roi_mask = df*0.0
    # Add all missing combinations with NaNs
    time_roi_mask = time_roi_mask.reindex(index=index, columns=columns)
    return time_mask, roi_mask, time_roi_mask


def df_epoch(df):
    '''Order DataFrame by epochs (epoch must be [first] index)'''
    ret = pd.DataFrame()
    for col in epochs.values:
        ret = ret.append(df.loc[[col],:])
    return ret


def load_files(mydir):
    '''Load files of the Losonczi group'''

    # Load files
    data = Bunch()
    data.experiment_traits = __format_experiment_traits(
        pd.read_hdf(os.path.join(mydir,'experiment_traits.h5'),key='table'))
    # Raw Ca-signal
    data.raw = pd.read_hdf(os.path.join(mydir,'raw_data.h5'),key='table').sort_index()
    data.raw.columns = pd.Index(data.raw.columns.values.astype(int), name='frame')
    # Filtered Ca-signal
    data.filtered = pd.read_hdf(os.path.join(mydir,'df_data.h5'),key='table').sort_index()
    data.filtered.columns = pd.Index(data.filtered.columns.values.astype(int), name='frame')
    # Spike-sorted
    data.transients = pd.read_hdf(os.path.join(mydir,'transients_data.h5'),key='table')
    # Licking behavior
    data.behavior = pd.read_hdf(os.path.join(mydir,'behavior_data.h5'),key='table')
    data.behavior.index.name='time'
    # Additional parameters (?)
    data.fluor = __load_fluor(mydir)

    # Add metadata
    data.max_nframe = data.raw.shape[1]
    data.FPS = int(np.floor(data.max_nframe/60.))
    if data.FPS not in [8, 30]:
        warnings.warn('FPS guess might be wrong.')
    data.event_frames = events*data.FPS
    data.trials = data.raw.index.levels[0]
    data.rois = data.raw.index.levels[1]
    data.roi_df = pd.DataFrame(data.rois, columns=['roi_id']
                    ).reset_index().rename(columns={'index':'idx'})
    data.mirow = pd.MultiIndex.from_product(
                (data.trials.values,data.rois.values),names=('time','roi_id'))
    data.micol = pd.MultiIndex.from_product(
                ('Spiking',np.array(range(0,data.max_nframe))),names=('','frame'))
    data.icol = pd.Index(np.array(range(0,data.max_nframe)),name='frame')

    data.time_mask, data.roi_mask, data.time_roi_mask = __create_mask(data.raw, data.mirow, data.icol)
    return data



### z-scoring ###

def pd_zscore_rows(df):
    '''z-score the rows of a DataFrame'''
    ret = df.copy()
    for idx, row in df.iterrows():
        ret.loc[idx,:] = (row - row.mean())/row.std(ddof=0)
    return ret

def nan_zscore(data):
    '''z-score list-like that may contain NaNs'''
    return (data-np.nanmean(data))/np.nanstd(data)

def nan_zscore_clip(data, clip_left=0, clip_right=-1):
    '''z-score list-like that may contain NaNs based on slice'''
    std = np.nanstd(data[clip_left:clip_right])
    if std>0:
        mea = np.nanmean(data[clip_left:clip_right])
        return (data-mea)/std
    else:
        return np.nan*np.ones(data.shape)

def pd_zscore_clip(df, clip_left=0, clip_right=-1, axis = 0):
    '''z-score the rows of a DataFrame based on slice'''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ret = df.apply(lambda x: nan_zscore_clip(x,clip_left,clip_right), axis=axis, raw=True)
    return ret

def pd_zscore_by_roi(df, clip_left=0, clip_right=-1, axis=None):
    '''z-score the rows of a special DataFrame with MultiIndex (trial, roi)
       based on a slice of columns grouping by the roi level of the index'''
    mea = df.iloc[:,clip_left:clip_right].stack().mean(level=1)
    std = df.iloc[:,clip_left:clip_right].stack().std(level=1)
    ret = df.sub(mea, axis='rows', level=1).divide(std, axis='rows', level=1)
    return ret


### data manipulation ###

def func_over_intervals(func, intervals, data):
    '''apply the same function to data blocks
       selected by [slice intervals) on the first axis'''
    axis=0
    shape = np.array(data.shape)
    n_ivs = len(intervals)-1
    shape[axis] = n_ivs
    ret = np.zeros(shape)
    for i in range(0,n_ivs):
        ret[i] = func(data[intervals[i]:intervals[i+1]])
    return tuple(ret)


def pd_aggr_col(df, pd_func, intervals, names):
    '''apply the same function to data blocks of DataFrame
       selected by [slice intervals) on the second axis'''
    if (len(names)+1 != len(intervals)):
        raise ValueError("sections and names are not matched")
    ser = [pd.DataFrame([], index=df.index)]
    for i in range(0,len(names)):
        ser.append(pd_func(df.iloc[:,intervals[i]:intervals[i+1]], axis=1).to_frame(name=names[i]))
    ret = pd.concat(ser, axis=1)
    return ret


def peri_event_avg(data, triggers, diameter=(-10, 10), allow=None, disable=None):
    '''Collect data in windows arond events in an event list'''
    window = np.arange(diameter[0],diameter[1])
    count=0
    ret = []
    for idx, weight in triggers.iteritems():
        experiment_id, frame = idx
        if (experiment_id in data.index) and ((allow is None) or (idx in allow)) and ((disable is None) or (idx not in disable)):
            tmp = data.loc[experiment_id,:].reindex(columns=frame+window)
            tmp.columns = pd.MultiIndex.from_product([count,window],names=['id','frame'])
            ret.append(tmp)
            count += 1
    if len(ret):
        ret = pd.concat(ret,axis=1)
        return ret, count
    else:
        return None, count


def get_gauss_window(time_range, rate):
    '''Gaussian window'''
    rate = float(rate)
    if type(time_range) is int:
        time_points = np.arange(0,time_range)-0.5*time_range
    elif len(time_range)==2:
        time_points = np.arange(time_range[0],time_range[-1])
    else:
        time_points = time_range
    decay = np.exp(-np.power(rate*time_points,2)/2.0)
    return decay/np.sum(decay)


def get_decay(time_range, rate):
    '''Exponentially decaying series'''
    rate = float(rate)
    if type(time_range) is int:
        time_points = np.arange(0,time_range)-0.5*time_range
    elif len(time_range)==2:
        time_points = np.arange(time_range[0],time_range[-1])
    else:
        time_points = time_range
    decay = np.exp(-np.abs(rate*time_points))
    return decay/np.sum(decay)


def rev_align(data, shape):
    '''Align for broadcast to shape matching axes from the beginning (opposed to numpy convention)'''
    data_dim = data.ndim
    req_dim = len(shape)
    new_axes = np.arange(data_dim,req_dim)
    # TODO: using np.reshape is more efficient
    ret = data
    for axis in new_axes:
        ret = np.expand_dims(data, axis=axis)
    return ret


def rev_broadcast(data, shape):
    '''Broadcast to shape matching axes from the beginning (opposed to numpy convention)'''
    ret = np.broadcast_to(rev_align(data,shape), shape)
    return ret


### Pattern atching ###

def match_pattern(data, pattern, std, decay, noise_level=0.01, detailed=False):
    '''Match pattern with decaying strength along time axis (rows). Use any number of columns.
    Pattern and std may be one (time points given, all columns equal) or two dimensional (matrix given).
    Noise_level can be 0 to 2 dimensional, if it is one dimensional then it is understood
    on the category axis (all rows equal), noise_level must be positive if there is any std==0.'''
    diff = data-rev_align(pattern, data.shape)
    scale = rev_align(decay, data.shape)/(noise_level+rev_align(std, data.shape))
    ret = np.nanmean(np.abs(diff*scale), axis=(0 if detailed else None))
    return -ret


def correlate_pattern(data, pattern, std, decay, noise_level=0.01, detailed=False):
    '''Multiply pattern with decaying strength along time axis (rows). Use any number of columns.
    Pattern and std may be one (time points given, all columns equal) or two dimensional (matrix given).
    Noise_level can be 0 to 2 dimensional, if it is one dimensional then it is understood
    on the category axis (all rows equal), noise_level must be positive if there is any std==0.'''
    # Note: this is not real correlation unless input is normalized
    corr = data*rev_align(pattern, data.shape)
    scale = rev_align(decay, data.shape)/(noise_level+rev_align(std, data.shape))
    ret = np.nanmean((corr*scale), axis=(0 if detailed else None))
    return ret


def rolling2D(df, func, window, min_periods=None, center=True):
    '''Slice a DataFrame along index (rows) to apply 2D function'''
    # This was a missing feature in pandas: one could previously correlate a single pattern
    # along a selected axis of a 2D DataFrame.
    window = int(window)
    if window<1:
        raise ValueError('window needs positive length')
    if min_periods is None:
        min_periods = window
    else:
        min_periods = int(min_periods)
    if min_periods<1:
        raise ValueError('min_periods needs to be positive')
    start = min_periods-window # first point of first window is start, available points evaluate to [0, min_periods)
    end = len(df)-min_periods # first point of last window is end, available points evaluate to [len-min_periods, len)
    if center:
        shift = int(np.floor(window/2))
    else:
        shift = 0
    first = start
    data = df.iloc[first:first+window,:]
    tmp = func(data)
    try:
        if len(tmp)==len(df.columns):
            ret = pd.DataFrame([], index=df.index, columns=df.columns)
        else:
            ret = pd.DataFrame([], index=df.index, columns=pd.Index(np.arange(0,len(tmp))))
    except:
        ret = pd.DataFrame([], index=df.index, columns=pd.Index([0]))
    ret.iloc[first+shift,:]=tmp
    for first in range(start+1,end+1):
        data = df.iloc[first:first+window,:]
        tmp = func(data)
        ret.iloc[first+shift,:]=tmp
    return ret


def search_pattern(df, triggers, trials, FPS, diam = (-3,3), decay_time=0.1, trigger_allow=None, trigger_disable=None, method='correlate'):
    '''deduce peri-event pattern based on triggers and search for similarities
       in the whole time series that is provided in (trial,roi) x (frames) format'''
    ret = []
    diam = int(FPS*diam[0]),int(FPS*diam[1])
    window = diam[1]-diam[0]
    decay = get_decay(window,1.0/(decay_time*FPS))
    dd, c = peri_event_avg(df, triggers, diameter=diam, allow=trigger_allow, disable=trigger_disable)
    p1 = dd.mean(axis=1, level=1).T.values
    s1 = dd.std(axis=1, level=1).T.values
    if method=='match':
        func = (lambda x: match_pattern(x.values,p1,s1,decay=decay))
    elif method=='correlate':
        p1 = p1 - np.nanmean(p1)
        func = (lambda x: correlate_pattern(x.values-x.mean().mean(),p1,s1,decay=decay))
    else:
        raise ValueError('Unaccepted method')
    for trial in trials:
        tmp = rolling2D(df.loc[trial,:].T,func,window,center=True).T
        tmp.index=[trial]
        ret.append(tmp)
    ret = pd.concat(ret)
    return ret.astype(float)


### Plotting specific to Losonczi Lab data ###

def plot_activity(df, et, FPS, grp = ['context','learning_epoch','port','puffed'],
                  name = 'Population activity (spiking)', ax = None, div=None, fill=None, alpha=0.05):
    '''Perform grouping and plot data into one subplot'''
    # NOTE: session_num is a string object therefore it is not included in the summation or averaging at the aggregation step of groupby
    # but the traits port and puffed are boolean and kept if uniform, so we get rid of them by conforming to the original index
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('xx-small')
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    if len(grp):
        gb = df.join(et,how='left').groupby(grp)
        res = gb.mean().reindex(columns=df.columns)
        try:
            std = gb.std().reindex(columns=df.columns)
        except:
            #C:\Program Files\Anaconda3\envs\py27\lib\site-packages\pandas\core\groupby.py in std(self, ddof)
            #   979         # todo, implement at cython level?
            #-->980         return np.sqrt(self.var(ddof=ddof))
            #AttributeError: 'float' object has no attribute 'sqrt'
            std = 0
        #count = df[[]].reset_index().drop_duplicates(['time']).set_index(['time']).join(et,how='left').groupby(grp).count()
        count = et.reindex(pd.Index(df.index.get_level_values('time').unique(), name='time')).groupby(grp).count()
        if (count.ndim>1):
            count = count.ix[:,0]
        for i in range(0,len(res)):
            val = res.values[i]
            try:
                err = std.values[i]
            except:
                # Same mysterious error
                err = 0
            pos = np.arange(0,len(val)) if div is None else div
            cnt = count.values[i]
            if fill=='std':
                f1 = ax.fill_between(pos, val-err, val+err, alpha=alpha, interpolate=False, edgecolor=None)
            elif fill=='err':
                f1 = ax.fill_between(pos, val-err/np.sqrt(cnt), val+err/np.sqrt(cnt), alpha=alpha, interpolate=False, edgecolor=None)
            l1 = ax.plot(pos, val, label=('%s: %d'%(res.index.values[i],cnt)))
            if fill is not None:
                f1.set_facecolors(l1[0].get_color())
                f1.set_edgecolors('none')
    else:
        res = df.mean(axis=0)
        val = res.values
        std = df.std(axis=0).values
        pos = np.arange(0,len(res)) if div is None else div
        if fill=='std':
            f1 = ax.fill_between(pos, val-err, val+err, alpha=alpha, interpolate=False, edgecolor=None)
        elif fill=='err':
            f1 = ax.fill_between(pos, val-err/np.sqrt(cnt), val+err/np.sqrt(cnt), alpha=alpha, interpolate=False, edgecolor=None)
        ax.plot(pos,res.values,label='whole popuation')
        if fill is not None:
            f1.set_facecolors(l1[0].get_color())
            f1.set_edgecolors('none')
    q = res.values.ravel()
    q = q[np.isfinite(q)]
    q = np.percentile(q[np.isfinite(q)],[1,99]) if len(q)>2 else np.array([0,1])
    ax.set_ylim(np.mean(q)+2*(q-np.mean(q)))
    ax.set_xlim(xmin=0)
    for i in range(0,len(events)):
        ax.axvline(x=events[i]*FPS, ymin=0.0, ymax = 1.0, linewidth=1, color='k')
    ax.set_xlabel('Camera frame')
    ax.set_ylabel(name)
    #leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=', '.join(grp)) # prop=fontP)
    #leg.get_title().set_fontsize('large')
    #leg.get_title().set_fontweight('bold')
    #ax.show()


# # Some experimenting
#
# def plot_data(df_spike, df_data, df_lick, grps = [[]], title=''):
#     from mpl_toolkits.axes_grid1 import host_subplot, AxesGrid
#     import mpl_toolkits.axisartist as AA
#     ncol = len(grps)
#     #fig, ax = plt.subplots(4, ncol, figsize=(6*ncol,13), sharex=True, squeeze=False)
#     fig = plt.figure(figsize=(6*ncol,13))
#     ax = np.ndarray((4,ncol),dtype=object)
#     #fig.tight_layout(pad=3, h_pad=3)
#     if len(title):
#         fig.suptitle(title, fontsize=16)
#     for i in range(0, ncol):
#         ax[0,i] = host_subplot(4, ncol, i*ncol+1)
#         ax[0,i].axis('off')
#         ax[1,i] = host_subplot(4, ncol, i*ncol+2)
#         plot_activity(df_spike,grps[i],"Spiking",ax=ax[1,i])
#         leg = ax[1,i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), title=', '.join(grps[i]))
#         leg.get_title().set_fontsize('large')
#         leg.get_title().set_fontweight('bold')
#         ax[2,i] = host_subplot(4, ncol, i*ncol+3)
#         plot_activity(df_data,grps[i],"Ca-level",ax=ax[2,i])
#         ax[2,i].legend_.remove()
#         ax[3,i] = host_subplot(4, ncol, i*ncol+4, axes_class=AA.Axes)
#         plot_activity(df_lick,grps[i],"Licking",ax=ax[3,i])
#         ax[3,i].legend_.remove()
#
#         par2 = ax[3,i]#.twiny()
#         #par2.set_visible(False)
#         new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#         par2.axis["bottom"] = new_fixed_axis(loc="bottom",
#                                         axes=par2,
#                                         offset=(0, -50))
#         par2.axis["bottom"].toggle(all=True)
#         par2.set_xlabel("Velocity")
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore', UserWarning)
#         fig.show()

def plot_data(df_spike, df_data, df_lick, et, FPS, grps = [[]], title='', div=None, fill=None):
    ncol = len(grps)
    nrow = 3 if df_lick is None else 4
    fig, ax = plt.subplots(nrow, ncol, figsize=(6*ncol,1+3*nrow), sharex=True, squeeze=False)
    fig.tight_layout(pad=3, h_pad=3)
    if len(title):
        fig.suptitle(title, fontsize=16)
    for i in range(0, ncol):
        ax[0,i].axis('off')
        plot_activity(df_spike, et, FPS, grps[i], "Spiking", ax=ax[1,i],div=div,fill=fill)
        leg = ax[1,i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), title=', '.join(grps[i]))
        leg.get_title().set_fontsize('large')
        leg.get_title().set_fontweight('bold')
        plot_activity(df_data, et, FPS, grps[i], "Ca-level", ax=ax[2,i],div=div,fill=fill)
        #ax[2,i].legend_.remove()
        if df_lick is not None:
            plot_activity(df_lick, et, FPS, grps[i], "Licking", ax=ax[3,i],div=div,fill=fill)
            #ax[3,i].legend_.remove()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        #fig.show()
        return fig

def plot_epochs(df_spike, df_data, df_lick, et, etc, FPS, grps = [[]], title='', div=None, fill=None):
    ncol = len(epochs)
    nrow = 3 if df_lick is None else 4
    fig, ax = plt.subplots(nrow, ncol, figsize=(6*ncol,1+3*nrow), sharex=True, squeeze=False)
    fig.tight_layout(pad=3, h_pad=3)
    if len(title):
        fig.suptitle(title, fontsize=16)
    for i in range(0, ncol):
        epoch = epochs.values[i]
        keys = etc[[]].reset_index()
        keys['learning_epoch'] = epoch
        sel = et.reset_index(drop=True).merge(keys, on=['context', 'port', 'puffed', 'learning_epoch'], how='inner')
        ax[0,i].set_title(epoch,y=0.8)
        ax[0,i].axis('off')
        plot_activity(df_spike.reindex(sel.loc[:,'timestr'],level=0), et, FPS, grps[0],"Spiking",ax=ax[1,i],div=div,fill=fill)
        leg = ax[1,i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), title=', '.join(grps[0]))
        leg.get_title().set_fontsize('large')
        leg.get_title().set_fontweight('bold')
        plot_activity(df_data.reindex(sel.loc[:,'timestr'],level=0), et, FPS, grps[0],"Ca-level",ax=ax[2,i],div=div,fill=fill)
        #ax[2,i].legend_.remove()
        if df_lick is not None:
            plot_activity(df_lick.reindex(sel.loc[:,'timestr'].rename('time')), et, FPS, grps[0],"Licking",ax=ax[3,i],div=div,fill=fill)
            #ax[3,i].legend_.remove()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        #fig.show()
        return fig


def draw_transients(ax, transients, experiment_id, FPS, roi_df):
    '''Plot transients with colored line and put a tic at the maxima'''
    import matplotlib
    ncolors = 10
    # Plot all neural units in this experiment
    try:
        firing = transients.loc[experiment_id,['start_frame', 'stop_frame', 'max_frame']].join(roi_df.set_index(['roi_id']), how='left')
        # Reshape things so that we have a sequence of:
        # [[(x0,y0),(x1,y1)],[(x0,y0),(x1,y1)],...]
        # based on http://stackoverflow.com/questions/17240694/python-how-to-plot-one-line-in-different-colors
        segments = firing[['start_frame', 'idx', 'stop_frame', 'idx']].values.reshape(-1,2,2)
        coll = matplotlib.collections.LineCollection(segments, cmap=plt.cm.rainbow)
        coll.set_array(firing['idx']%ncolors)
        if len(firing):
            #ax.plot(firing[['start_frame', 'stop_frame']].T,firing[['idx', 'idx']].T,c=colors[firing['idx']])
            ax.add_collection(coll)
            ax.autoscale_view()
        if len(firing):
            ax.plot(firing['max_frame'].T,firing['idx'].T,'|',ms=5,c='k',label='spikes')
            #xlim, ylim = ax.get_xlim(), ax.get_ylim()
            #ax.scatter(firing['max_frame'].T,firing['idx'].T,s=5,c='k',marker='|')
            #ax.set_xlim(xlim), ax.set_ylim(ylim)
    except:
        pass
    for i in range(0,len(events)):
        ax.axvline(x=events[i]*FPS, ymin=0.0, ymax = 1.0, linewidth=1, color='k')
    ax.set_title('Transient peaks and durations')
    ax.set_xlabel('Camera frame')
    ax.set_ylabel('Unit ID')

def draw_levels(ax, data, experiment_id, FPS, roi_df, zoom=0.5, dist=1.0):
    '''Plot transients with colored line and put a tic at the maxima'''
    # Plot all neural units in this experiment
    try:
        firing = (zoom*data.loc[experiment_id,:]).add(dist*roi_df.set_index(['roi_id']).loc[:,'idx'],axis=0)
        if len(firing):
            ax.plot(firing.T)#,c=colors)
    except:
        pass
    for i in range(0,len(events)):
        ax.axvline(x=events[i]*FPS, ymin=0.0, ymax = 1.0, linewidth=1, color='k')
    ax.set_title('Raw Ca-levels')
    ax.set_xlabel('Camera frame')
    ax.set_ylabel('Unit ID')

def draw_spiking_nan(ax, spiking, experiment_id, roi_df):
    '''Mark unavailable data with a gray dot'''
    # Plot all neural units in this experiment
    try:
        firing = spiking.loc[experiment_id,:].join(roi_df.set_index(['roi_id']), how='left')
        firing.columns.name='frame'
        firing = firing.set_index('idx').stack(dropna=False)
        firing = firing[firing.isnull()]
        firing = firing.reset_index()
        if len(firing):
            #ax.scatter(firing.loc[:,'frame'],firing.loc[:,'idx'],s=1,c='lightgray',marker='.',label='missing')
            ax.plot(firing.loc[:,'frame'],firing.loc[:,'idx'],'.',ms=1,c='lightgray',label='missing')
    except:
        pass

def draw_conditions(ax, conditions, experiment_id, FPS, height=20, loc='lower center', screen_width=1.0, fontsize=24, cw=None):
    '''Draw a table and write experimental conditions into it'''
    import matplotlib
    a = conditions.loc[[experiment_id],['learning_epoch','context','port','puffed','session_num','day_num']]
    if cw is None:
        cw = np.concatenate((durations[1:]*FPS,np.array([0.5,0.5])*(ax.get_xlim()[1]-events[-1]*FPS)))

    c = a.copy()
    c.loc[:,:]='lightblue' if any(a['port'].isin(['W+',True])) else 'white'
    replacement = [('context', 'CS-', 'lightgreen'), ('context', 'CS+', 'lightcoral'),
                   ('context', 'Baseline', 'lightblue'),
                   ('port', 'W+', 'lightblue'), ('puffed', 'A+', 'yellow'),
                   ('port', True, 'lightblue'), ('puffed', True, 'yellow')]
    for label, value, color in replacement:
        c.loc[a[label]==value,label]=color

    ylim = ax.get_ylim()
    #tab = pd.tools.plotting.table(ax, a, loc='lower center', fontsize=24, colWidths=cw/np.sum(cw))
    tab = matplotlib.table.table(ax, cellText=a.values,
                                   #rowLabels=rowLabels, colLabels=colLabels,
                            loc=loc, fontsize=24, colWidths=cw/np.sum(cw), bbox=[0,0,screen_width,height/(ylim[1]-ylim[0])], cellLoc='center', cellColours=c.values)
    # fontsize keyword is accepted but seems ineffective
    #tab.set_fontsize(fontsize)
    for key, cell in tab.get_celld().items():
        cell.set_linewidth(0)
        cell.set_fontsize(fontsize)


def draw_triggers(ax, triggers, experiment_id, pos=0, ls='x', c='b', ms=8):
    '''Plot trigger events'''
    try:
        if type(triggers) is not list:
            triggers=[triggers]
        for i, trig in enumerate(triggers):
            x = trig.loc[experiment_id].index.values
            x = x.reshape((1,-1))
            if x.shape[1]>0:
                ls1 = ls[i] if type(ls) is list else ls
                c1 = c[i] if type(c) is list else c
                ms1 = ms[i] if type(ms) is list else ms
                ax.plot(x, pos, ls1, c=c1, ms=ms1)
    except:
        pass

def draw_behavior(ax, licks, experiment_id, FPS):
    '''Plot individual licks'''
    try:
        i=-5
        licking = np.array(licks.loc[experiment_id,['start_time', 'stop_time']])*FPS
        if len(licking):
            ax.plot(licking.T,i*np.ones_like(licking.T),c='b')
        licking = np.array(licks.loc[experiment_id,['start_time', 'stop_time']].mean(axis=1))*FPS
        if len(licking):
            ax.plot(licking,i*np.ones_like(licking),'o',ms=5,c='k')
    except:
        pass

def draw_licking(ax, licking, experiment_id, pos=-20, zoom=1.0, c='b', threshold=None, label=None):
    '''Plot licking rate or any other single time series'''
    try:
        ax.axhline(y=pos, xmin=0.0, xmax = 1.0, linewidth=1, color='k')
        if threshold is not None:
            try:
                threshold = [float(threshold)]
            except:
                pass
            for thr in threshold:
                ax.axhline(y=thr*zoom+pos,c='lightgray')
        licking = licking.loc[experiment_id,:].values
        if len(licking):
            ax.plot(licking*zoom+pos,c=c,label=label)
    except:
        pass

def draw_population(ax, data, experiment_id, pos=-20, zoom=10.0, c='r', threshold=None, label=None):
    '''Plot population activity from individual signals'''
    try:
        ax.axhline(y=pos, xmin=0.0, xmax = 1.0, linewidth=1, color='k')
        if threshold is not None:
            try:
                threshold = [float(threshold)]
            except:
                pass
            for thr in threshold:
                ax.axhline(y=thr*zoom+pos,c='lightgray')
        data = data.loc[experiment_id,:].mean(axis=0)
        if len(data):
            ax.plot(data*zoom+pos,c=c,label=label)
    except:
        pass

def show_peri_event1(ax, df, title=None, pos=-15, zoom=10.0, vmin=None, vmax=None):
    '''Plot df using matshow'''
    extent = np.min(df.columns.values)-0.5, np.max(df.columns.values)+0.5, -0.5, len(df)+0.5
    ax.set_xlim(extent[0:2])
    ax.set_ylim((pos-zoom,extent[3]))
    ret = ax.matshow(df, origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    ax.axhline(y=pos,xmin=0.0,xmax=1.0,c='gray')
    ax.axvline(x=0,ymin=0.0,ymax=1.0,c='gray')
    ax.plot(zoom*df.mean(axis=0)+pos)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('$\Delta$frame')
    ax.set_ylabel('Unit ID')
    ax.set_xlim(extent[0:2])
    ax.set_ylim((pos-zoom,extent[3]))
    if title is not None:
        ax.set_title(title)
    return ret
    
def show_peri_event2(ax, df_mean, df_std, title=None, pos=-15, zoom=10.0, vmin=None, vmax=None):
    import matlab_tools as mt
    extent = np.min(df_mean.columns.values)-0.5, np.max(df_mean.columns.values)+0.5, -0.5, len(df_mean)+0.5
    '''Plot mean and std using color and lightness-encoding'''
    ax.set_xlim(extent[0:2])
    ax.set_ylim((pos-zoom,extent[3]))
    #img = mt.hls_matrix(mt.crop_series(0.4-0.5*df_mean.T.values,(0,0.8)),mt.crop_series(0.5-0.5*df_std.T.values,(0,1)),0.6)
    img = mt.hls_matrix(mt.crop_series(0.2-0.25*df_mean.T.values,(0,0.8)),mt.crop_series(0.5-0.25*df_std.T.values,(0,1)),0.6)
    ret = ax.imshow(img,interpolation='none',origin='lower',aspect='auto', extent=extent)
    ax.axhline(y=pos,xmin=0.0,xmax=1.0,c='gray')
    ax.axvline(x=0,ymin=0.0,ymax=1.0,c='gray')
    ax.plot(zoom*df_mean.mean(axis=0)+pos)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('$\Delta$frame')
    ax.set_ylabel('Unit ID')
    ax.set_xlim(extent[0:2])
    ax.set_ylim((pos-zoom,extent[3]))
    if title is not None:
        ax.set_title(title)
    return ret
    
def plot_peri_collection(collection, title=None, combine=True):
    '''Plot a colection of peri-event activities provided in a list'''
    max_cols = 10
    num_plots = len(collection) * (1 if combine else 2)
    num_rows = int(np.ceil(num_plots/float(max_cols)))
    num_cols = max_cols if num_rows>1 else num_plots
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(2*num_cols+2,12*num_rows), sharex=True, sharey=True, squeeze=False)
    ax = np.ravel(ax)
    fig.tight_layout(rect=(0,0,0.9,0.9), w_pad=2, h_pad=8)
    gradient = np.linspace(-1, 1, 256)
    gradient = pd.DataFrame(np.vstack((gradient, gradient)).T, columns=[2,3])
    cax1 = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cax2 = fig.add_axes([0.96, 0.1, 0.02, 0.8])
    if combine:
        show_peri_event2(cax1, gradient, gradient*0, 'Mean', vmin=-1, vmax=1)
        show_peri_event2(cax2, gradient*0, gradient+1, 'Std', vmin=-1, vmax=1)
    else:
        show_peri_event1(cax1, gradient, 'Mean', vmin=-1, vmax=1)
        show_peri_event1(cax2, gradient+1, 'Stdev', vmin=0, vmax=2)
    cax1.set_ylabel(''); cax1.set_yticks([0,128,256]); cax1.set_yticklabels([-1,0,1])
    cax2.set_ylabel(''); cax2.set_yticks([0,128,256]); cax2.set_yticklabels([0,1,2])
    cax1.set_xlabel(''); cax1.set_xticks([])
    cax2.set_xlabel(''); cax2.set_xticks([])
    fig.suptitle(title,fontsize=16)
    
    for i, (df, index, trig, allow, disable, title) in enumerate(collection):
        dd, c = peri_event_avg(df, trig, allow=allow, disable=disable)
        if c:
            if index is not None:
                dd = dd.reindex(index)
            if combine:
                show_peri_event2(ax[i], dd.mean(axis=1, level=1), dd.std(axis=1, level=1), '%s: %d'%(title,c), vmin=-1, vmax=1)
            else:
                show_peri_event1(ax[2*i], dd.mean(axis=1, level=1), '%s: %d'%(title,c), vmin=-1, vmax=1)
                show_peri_event1(ax[2*i+1], dd.std(axis=1, level=1), 'Stdev', vmin=0, vmax=2)

    return fig