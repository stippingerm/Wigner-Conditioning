# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:47:42 2016

@author: Marcell
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Set event lengths
phases=['Ready','CS','Trace','US','End']
durations=np.array([0,10,20,15,5])
events=np.cumsum(durations).astype(int)

# Set date formate
dtformat = '%Y-%m-%d-%Hh%Mm%Ss'
sessionbreak = np.timedelta64(8,'h')

epochs = pd.CategoricalIndex(['Pre-Learning', 'Learning', 'Post-Learning'], ordered=True)

display_learning = ['learning_epoch','context','puffed','port']
sort_learning = ['learning_epoch','context','port','puffed']
sort_context = ['context','learning_epoch','port','puffed']

def df_epoch(df):
    '''Order DataFrame by epochs (epoch must be [first] index)'''
    ret = pd.DataFrame()
    for col in epochs.values:
        ret = ret.append(df.loc[[col],:])
    return ret
    
class Bunch:
    '''Dot-access container'''
    def __init__(self, **kw):
        for name in kw:
            setattr(self, name, kw[name])
    def __repr__(self):
        return self.__dict__.__repr__()
    def __str__(self):
        return self.__dict__.__str__()

def __format_experiment_traits(df):
    '''Clarify traits'''
    from datetime import datetime as dt
    et = df.rename(columns={'licking':'port','time':'timestr'})
    et['port'] = et['port'].apply(lambda x: 'W+' if x else 'W-')
    et['puffed'] = et['puffed'].apply(lambda x: 'A+' if x else 'A-')
    et['session_num'] = et['session_num'].astype(int)
    et['time'] = et['timestr'].apply(lambda t: dt.strptime(t, dtformat))
    leapaday = (et['time'].values[1:]-et['time'].values[:-1]) > sessionbreak
    et['day_leap'] = np.append([True],leapaday)
    et['day_num'] = np.cumsum(np.append([0],leapaday.astype(int)))
    return et

def __load_fluor(mydir):    
    '''Load the undocumented dict called fluor'''
    import pickle
    
    # Some undocumented info about the experiments
    pkl_file = open(os.path.join(mydir,'frame_fluor.pkl'), 'rb')
    
    # Python 2.7
    ret = pickle.load(pkl_file)
    
    # Python 3.5
    #u = pickle._Unpickler(pkl_file)
    #u.encoding = 'latin1'
    #ret = u.load()
    
    return ret
        

def __create_mask(df, index=None, columns=None, threshold=0.9):
    time_mask = df.reset_index().groupby(['time']).count()
    time_mask = time_mask.drop(['roi_id'],axis=1).subtract(threshold*time_mask['roi_id'],axis=0)<0
    roi_mask = df.reset_index().groupby(['roi_id']).count()
    roi_mask = roi_mask.drop(['time'],axis=1).subtract(threshold*roi_mask['time'],axis=0)<0
    time_roi_mask = df*0.0
    time_roi_mask = time_roi_mask.reindex(index=index, columns=columns)
    return time_mask, roi_mask, time_roi_mask

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
        warnings.warn('FPS might be wrong.')
    data.events = events
    data.event_frames = events*data.FPS
    data.trials = data.raw.index.levels[0]
    data.rois = data.raw.index.levels[1]
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

#def pd_zscore_by_roi(df, clip_left=0, clip_right=-1, axis=None):
#    '''z-score the rows of a special DataFrame with index (trial, roi)
#       based on a column slice'''
#    ret = df.reset_index().set_index(['roi_id','time'])
#    trf = lambda x: nan_zscore_clip(x.T,clip_left,clip_right).T
#    for roi in ret.index.levels[0]:
#        ret.loc[roi,:] = trf(ret.loc[roi,:].as_matrix())
#    return ret.reset_index().set_index(['time','roi_id'])
    
def pd_zscore_by_roi(df, clip_left=0, clip_right=-1, axis=None):
    '''z-score the rows of a special DataFrame with index (trial, roi)
       based on a column slice'''
    mea = df.iloc[:,clip_left:clip_right].stack().mean(level=1)
    std = df.iloc[:,clip_left:clip_right].stack().std(level=1)
    ret = df.sub(mea, axis='rows', level=1).divide(std, axis='rows', level=1)
    return ret
    



### data manipulation ###
def func_over_intervals(func, intervals, data, axis=0):
    shape = np.array(data.shape)
    n_ivs = len(intervals)-1
    shape[axis] = n_ivs
    ret = np.zeros(shape)
    for i in range(0,n_ivs):
        ret[i] = func(data[intervals[i]:intervals[i+1]])
    return tuple(ret)

# def pd_aggr(df, func, colnames, axis=1, raw=True):
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore', RuntimeWarning)
#         ret = df.apply(func, axis=axis, raw=raw)
#         ret = pd.DataFrame(ret.tolist(), columns=colnames, index=ret.index)
#     return ret

def pd_aggr_col(df, pd_func, sections, names):
    if (len(names)+1 != len(sections)):
        raise ValueError("sections and names are not matched")
    ser = [pd.DataFrame([], index=df.index)]
    for i in range(0,len(names)):
        ser.append(pd_func(df.iloc[:,sections[i]:sections[i+1]], axis=1).to_frame(name=names[i]))
    ret = pd.concat(ser, axis=1)
    return ret


### Plotting ###
    
def plot_activity(df, et, FPS, grp = ['context','learning_epoch','port','puffed'],
                  name = 'Population activity (spiking)', ax = None, div=None):
    # NOTE: session_num is a string object therefore it is not included in the summation or averaging at the aggregation step of groupby
    # but the traits port and puffed are boolean and kept if uniform, so we get rid of them by conforming to the original index
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('xx-small')
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    if len(grp):
        res = df.join(et,how='left').groupby(grp).mean().reindex(columns=df.columns)
        count = df[[]].reset_index().drop_duplicates(['time']).set_index(['time']).join(et,how='left').groupby(grp).count()
        if (count.ndim>1):
            count = count.ix[:,0]
        for i in range(0,len(res)):
            if div is None:
                ax.plot(res.values[i],label=('%s: %d'%(res.index.values[i],count.values[i])))
            else:
                ax.plot(div,res.values[i],label=('%s: %d'%(res.index.values[i],count.values[i])))
    else:
        res = df.mean(axis=0)
        if div is None:
            ax.plot(res.values,label='whole popuation')
        else:
            ax.plot(div,res.values,label='whole popuation')
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
        
def plot_data(df_spike, df_data, df_lick, et, FPS, grps = [[]], title='', div=None):
    ncol = len(grps)
    nrow = 3 if df_lick is None else 4
    fig, ax = plt.subplots(nrow, ncol, figsize=(6*ncol,1+3*nrow), sharex=True, squeeze=False)
    fig.tight_layout(pad=3, h_pad=3)
    if len(title):
        fig.suptitle(title, fontsize=16)
    for i in range(0, ncol):
        ax[0,i].axis('off')
        plot_activity(df_spike, et, FPS, grps[i], "Spiking", ax=ax[1,i],div=div)
        leg = ax[1,i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), title=', '.join(grps[i]))
        leg.get_title().set_fontsize('large')
        leg.get_title().set_fontweight('bold')
        plot_activity(df_data, et, FPS, grps[i], "Ca-level", ax=ax[2,i],div=div)
        #ax[2,i].legend_.remove()
        if df_lick is not None:
            plot_activity(df_lick, et, FPS, grps[i], "Licking", ax=ax[3,i],div=div)
            #ax[3,i].legend_.remove()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        #fig.show()
        return fig
        
def plot_epochs(df_spike, df_data, df_lick, et, etc, FPS, grps = [[]], title='', div=None):
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
        plot_activity(df_spike.reindex(sel.loc[:,'timestr'],level=0), et, FPS, grps[0],"Spiking",ax=ax[1,i],div=div)
        leg = ax[1,i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), title=', '.join(grps[0]))
        leg.get_title().set_fontsize('large')
        leg.get_title().set_fontweight('bold')
        plot_activity(df_data.reindex(sel.loc[:,'timestr'],level=0), et, FPS, grps[0],"Ca-level",ax=ax[2,i],div=div)
        #ax[2,i].legend_.remove()
        if df_lick is not None:
            plot_activity(df_lick.reindex(sel.loc[:,'timestr'].rename('time')), et, FPS, grps[0],"Licking",ax=ax[3,i],div=div)
            #ax[3,i].legend_.remove()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        #fig.show()
        return fig