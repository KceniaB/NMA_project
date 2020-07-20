import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import platform
import os

"""
Class for all the functions from Samual, Kcenia and Max NMA Projekt

- put in finalized functions - production ready
- pay close attantion on documentation and help files
"""



class NMA_project():
    def __init__(self, main_folder):
        self.folder=main_folder
        self.subfolders, self.all_sessions = self.get_available_session()
        # load first session
        self.std_session = self.load_session(self.subfolders[0])
    
    #  find all available sessions
    def get_available_session(self):
        subfolders = [ f.path for f in os.scandir(self.folder) if f.is_dir() ]
        sessions = [folder_.split('/')[-1] for folder_ in subfolders]
        return subfolders, sessions

    # load session from given sesion folder
    def load_session(self, folder):
        """
        load all files from given folder to dict
        return: (session::dictionary with key as filenames,
                    important dict element: 
                        clusters_df::dataframe with all clustes and infos about clusters and spikes for clusters,
                        trials_df:: dataframe with all trials and events of trials
                 )
        """
        # load session files ===========================================
        from os import listdir
        from os.path import isfile, join
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        names = ['_'.join(f.split('.')[:-1]) for f in files]
        session = dict()
        for (file_, name_) in zip(files, names):
            if file_.split('.')[-1] == 'npy':
                session[name_] = np.load(folder+'/'+file_)
            if file_.split('.')[-1] == 'tsv':
                session[name_] = pd.read_table(folder+'/'+file_)

        
        # pars probe infos ===========================================
        # load channel brain location infos
        brain = pd.DataFrame(session['channels_brainLocation'])
        # load channel probes
        site = pd.DataFrame(session['channels_sitePositions'], columns=['channel 0', 'channel 1'])
        # merge with channles_df
        channels_df = pd.merge(brain, site, how='inner', left_index=True, right_index=True)
        # load probe, row and site
        meta = pd.DataFrame({'probe':session['channels_probe'][:,0],
                                    'raw row':session['channels_rawRow'][:,0],
                                    'channels_site':session['channels_site'][:,0] })
        # merge metha with channel_df
        channels_df = pd.merge(channels_df, meta, how='inner', left_index=True, right_index=True)
        # add channels_df to session dictionary
        session['channels_df']=channels_df

        # pars spikes info =================================
        # create spike_df tataframe, with each spike time and the cluster it belongs to
        spikes_df = pd.DataFrame( { 'cluster':session['spikes_clusters'][:,0], 'spike_times': session['spikes_times'][:,0] }, ) 
        # add spikes to session dictionary
        session['spikes_df']=spikes_df

        # pars clusters info ===============================
        # create cluster dataframe information and spikes for each cluster
        clusters = np.unique(session['spikes_clusters'])
        # create number of spikes, phy2 manual cluster, 
        # parse phy annotation
        clusters_df = pd.DataFrame({'lable':session['clusters__phy_annotation'][:,0], 'probe':session['clusters_probes'][:,0], 'peak channel':session['clusters_peakChannel'][:,0]}, index=clusters )
        # pars labels in strings 1 = good 3 = mua
        #clusters_df['lable'] = clusters_df['lable'].apply( lambda lable: 'good' if lable==3 else ('mua' if lable==1 else 'bad') )

        # create spikes colum with spiketimes
        spk = pd.DataFrame( {'spikes':np.zeros(len(clusters), dtype=object)}, index=clusters )
        for group, frame in spikes_df.groupby('cluster'):
            spk['spikes'][group] = frame['spike_times'].values

        #merge spike column with clusters_df
        clusters_df = pd.merge(clusters_df, spk, how='right', left_index=True, right_index=True)
        # set index name
        clusters_df.index.name='cluster'
        # number of spikes per cluster
        clusters_df['total spikes'] = clusters_df['spikes'].apply(lambda row: len(row) )

        # pars channel info for cluster


        # add clusters_df to session dictionary
        session['clusters_df'] = clusters_df


        # pars trials infos ===========================================


        return session