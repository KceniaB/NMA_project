import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import platform
import os
from os import path

class SpikeEDA():
    """[library to perfrom exploratory data analyis on spikes]
    """    
    def __init__(self, data, session, main_folder, params):
        """[initialize session analyse object]

        Args:
            data ([toubple]): [(spikes_df, clusters_df, trials_df)]
            session ([string]): [session folder]
            main_folder ([type]): [description]
            params ([dict]): [dictionary with all necessary infromation
                                params['sampling_rate] ([int]): [sampling rate for spikes]
                            ]
        """        
        self.folder = main_folder
        self.spikes_df, self.clusters_df, self.trials_df = data
        self.session = session
        # load all parameters
        if 'sampling_rate' in params:
            self.sampling_rate = params['sampling_rate']
        else:
            self.sampling_rate = 20000


 # Helper Functions EDA =================================================================================================
    # find spikes between
    def get_spikes_for_trial(self, array, start, stop):
        '''
        params: array = numpy array (N,1) with values to check against
                start, stop = value to find all values in array between
        return: valus in array between start and stop
        '''
        ar = array[np.logical_and(array >= start, array <= stop)]
        if ar.size > 0:
            ar = ar[:] - ar[0]
        return ar






 # Plotting ==========================================================================================================
    # event plot for all spikes
    def plt_spike_train(self, cluster, spikes, trials_df, params=dict()):
        """[generate spike event plot for given spikes and given trial start and end]

        Args:
            cluster ([int]): [selected neuron]
            spikes ([numpy array]): [spikes for neuron to plot]
            trials_df ([pandas data frame]): [format: index=trial, col1=start of trial, col2=stop of trial]
            params ([dict]): [optional, default = empty, params['brain_region' = brain region of cluster] ]

        Returns:
            [type]: [description]
        """        
        # initialize plot
        fig, ax = plt.subplots()
        # initialize list with spikes per trial
        spikes_trials = []
        # get spikes for each trial
        for row in trials_df.index:
            start = trials_df.iloc[row, 0]
            stop = trials_df.iloc[row, 1]
            spk = self.get_spikes_for_trial(spikes, start, stop)
            #if len(spk)>0:
            spikes_trials.append(spk)
        # plot spikes
        ax.eventplot(spikes_trials, color=".2")
        # set title and axis labels
        if 'brain_region' in params:
            ax.set_title(f"Spikes for Cluster {cluster}, Brain Region: {params['brain_region']}")
        ax.set_title(f"Spikes for Cluster {cluster}")
        ax.set_xlabel(f"Sampling Points [{self.sampling_rate/1000}kHz]")
        ax.set_ylabel('Trial')
        index = trials_df.index[0::10]
        ax.set_yticks(index - index[0])
        ax.set_yticklabels(index)
        return ax, fig

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        plt.show

    # plot spike trains araound event with histogram for spike time and trial
    def plt_spike_train_hist_bar(self, cluster, event, window, trials_df, spikes_ar, fig=None, ax=[None, None, None], title=None):
        """[summary]

        Args:
            cluster ([int]): [neuron to plot]
            event ([string]): [event name]
            window ([int]): [window to plot spikes: event_time-window < spikes < event_time+window]
            trials_df ([dataframe]): [event times to plot for, must be only column in dataframe]
            spikes_ar ([numpy array]): [array with all spikes of neuron to plot]
            fig ([plt.subfig figure], optional): [description]. Defaults to None.
            ax (plt.subfig axis, optional): [description]. Defaults to [None, None, None].
            title ([string], optional): [description]. Defaults to None.

        Returns:
            fig [plt.subfig figure]: [description]
            (ax1, ax2, ax3) [plt.subfig axis]: [description]

        """        
        # create fig, gird and axis ===============
        if any(i==None for i in ax)or fig==None:
            #create figure with shape
            fig = plt.figure(figsize=(6,5))
            # create gridspecs
            gs = fig.add_gridspec(2, 3,  hspace=0, wspace=0)
            # create axis for hist spike train
            ax1 = fig.add_subplot(gs[0, :2])
            ax2 = fig.add_subplot(gs[1, :2])
            ax2.get_shared_x_axes().join(ax1, ax2)
            # create axis for trial hist
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.get_shared_y_axes().join(ax1, ax3)
        else:
            ax1, ax2, ax3 = ax
        
        # loop that iterats trough all trials
        y = 0
        # array to store spike count for each trial
        hist_tr = np.empty([0,])    
        # list to store spikes for time bin in for eventplot and histogram
        #spk_ar = np.empty((len(trials_df),1), dtype=object)
        spk_ls = []


        ##spike train plot ========================
        # main loop over each trial
        for row in trials_df.index:
            # derive spike times in range delta around event time for trial
            ar = spikes_ar[( ( spikes_ar >= (trials_df[row] - window) ) & ( spikes_ar <= (trials_df[row] + window) ) )]
            ar = ar - trials_df[row]
            # ad spike count to hist_tr for row
            hist_tr = np.append(hist_tr, ar.size)
            # add to histogram array
            spk_ls.append(ar.flatten().tolist())

        # plot eventplot
        ax1.eventplot(spk_ls, color=".2")
        ## draw red line at event
        ax1.axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # spike train y lable
        ax1.set_ylabel('Trial')
        ## set y axis 1. plot
        # set ticks
        step = trials_df.index.size/5
        start = 0
        stop = trials_df.index.size+step/2
        ax1.set_yticks(np.arange(start, stop, step).astype(int))
        # set tick labels
        stop = trials_df.index.size
        label = trials_df.index.values[np.arange(start, stop, step).astype(int)]
        label = np.append(label, trials_df.index.values[-1])
        ax1.set_yticklabels(label)
        # set y limits 1. plot
        ax1.set_ylim([0, stop])
        ##labels
        # trun x labels inside
        ax1.tick_params(axis="x",direction="in")
        # turn of labels on shared x axis only ticks
        plt.setp(ax1.get_xticklabels(), visible=False)
        # write event
        ax1.set_title(event, color='red', fontsize=8)

        ## plot histogram spikes ===========================
        num_bins = 60
        # flatten list of sikes for histogram
        flattened = [val for sublist in spk_ls for val in sublist]
        # draw histogram
        ax2.hist(flattened, bins=num_bins, color="tab:blue")
        # draw red line at event
        ax2.axvline(x=0,ymin=0,ymax=1,c="red",linewidth=0.5)
        # naming y axis
        ax2.set_ylabel('Spike Count')
        # set x ticks
        step = window/4
        start = -window
        stop = window+(step/2)
        x_ticks = np.arange(start, stop, step)
        ax2.set_xticks(x_ticks)
        # set x ticks labels to seconds
        # set x limits
        ax2.set_xlim([-window, window])
        if window > 1000:
            window = window/1000
        step = window/4
        start = -window
        stop = window+(step/2)
        x_labels = np.arange(start, stop, step)
        ax2.set_xticklabels(x_labels)
        # set ticks top and bottom
        ax2.tick_params(axis='x', bottom=True, top=True)


        ## plot histogram trials =================================
        pos = np.arange(0, len(hist_tr))
        # invert axis
        ax3.invert_xaxis()
        # remove ticks
        ax3.set_yticks([])

        ## plot histogram
        ax3 = plt.barh(pos, hist_tr.reshape(hist_tr.size), height=1.0, color='lightgray')

        # name main title
        #ax.set_title('Spikes for Cluster 1')
        if title != None:
            event = title
        fig.suptitle(f"Spikes for Cluster: {cluster} at Event: {event}")
        plt.xlabel('Position [ms]')
        
        return fig, (ax1, ax2, ax3)



 # Report Generation ===================================================================================================




