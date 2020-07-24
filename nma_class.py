import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import platform
import os, csv
from numba import jit
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d
from sklearn.decomposition import TruncatedSVD


"""
Class for all the functions from Samuel, Kcenia and Max NMA Projekt

- put in finalized functions - production ready
- pay close attantion on documentation and help files
"""



class NMA_project():
    def __init__(self, main_folder, dt=1/1000, dT=2.5, T0=0.5):
        self.dt = dt
        self.dT = dT
        self.T0 = T0
        self.regions, self.brain_groups = self.get_brain_region()
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
        files = [f for f in files if f[:2]!='._'] #Samuel added this to deal with weird bug
        names = ['_'.join(f.split('.')[:-1]) for f in files]
        session = dict()
        for (file_, name_) in zip(files, names):
            if file_.split('.')[-1] == 'npy':
                session[name_] = np.load(join(folder,file_))
            if file_.split('.')[-1] == 'tsv':
                session[name_] = pd.read_table(join(folder,file_))

        # Load Pachitariu Session Data ===========================================================================
        # good cells and brain regions
        good_cells, brain_region, br = self.get_good_cells(folder)
        # event types
        response, vis_right, vis_left, feedback_type = self.get_event_types(folder)
        # event timing
        response_times, visual_times, rsp, gocue, feedback_time = self.get_event_times(folder)   
        # get passive trials
        vis_times_p, vis_right_p, vis_left_p = self.get_passive(folder)
        visual_times = np.vstack((visual_times, vis_times_p))
        vis_right = np.hstack((vis_right, vis_right_p))
        vis_left  = np.hstack((vis_left, vis_left_p))
        # wheel traces
        stimes, sclust    = self.get_spikes(folder)
        # only care about spikes during trials
        wheel, wheel_times = self.get_wheel(folder)
        # load the pupil
        pup, xy, pup_times = self.get_pup(folder)
        # load the LFP
        #L, ba_lfp = self.get_LFP(folder, br, visual_times-T0, dT, dt, T0)
        # trials loader
        S  = self.psth(stimes, sclust,   visual_times-self.T0, self.dT, self.dt)
        # wheel trials
        W = self.wpsth(wheel, wheel_times,   visual_times-self.T0, self.dT, self.dt)
        # pupil loader
        P = self.ppsth(pup, pup_times,   visual_times-self.T0, self.dT, self.dt)
        # add spike waveform information
        twav, w, u = self.get_waves(folder)
        good_cells = good_cells * (np.mean(S, axis=(1,2))>0)
        S  = S[good_cells].astype('int8') 
    
        # pars probe infos ============================================================================
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

        # pars spikes info ============================================================================
        # create spike_df tataframe, with each spike time and the cluster it belongs to
        spikes_df = pd.DataFrame( { 'cluster':session['spikes_clusters'][:,0], 'spike_times': session['spikes_times'][:,0] }, ) 
        # add spikes to session dictionary
        session['spikes_df']=spikes_df

        # pars clusters info ============================================================================
        # create cluster dataframe information and spikes for each cluster
        clusters = np.unique(session['spikes_clusters'])
        # create number of spikes, phy2 manual cluster, 
        # leave out:  'probe':session['clusters_probes'][:,0], 'peak channel':session['clusters_peakChannel'][:,0]
        clusters_df = pd.DataFrame({'lable':session['clusters__phy_annotation'][:,0]}, index=clusters )
        # pars labels in strings 1 = good 3 = mua
        clusters_df['lable'] = clusters_df['lable'].apply( lambda lable: 'good' if lable==2 else ('mua' if lable==1 else 'bad') )

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
        # add recording position
        clusters_df['recording area'] = brain_region

        # add clusters_df to session dictionary
        session['clusters_df'] = clusters_df


        # pars trials infos ===========================================================================
        #total number of trials
        ntrials = session['trials_included'].shape[0]
        #number of active engaged trials
        ntrials_active = len(response)
        # create boolean array to denote active trials
        active = np.zeros(ntrials)
        active[:ntrials_active] = 1
        # create trials dataframe
        trials_df = pd.DataFrame({'included':session['trials_included'][:,0],
                                'active': active,
                                'repetition number':session['trials_repNum'][:,0],
                                # times of events
                                'go cue':session['trials_goCue_times'][:,0],
                                'stim time':session['trials_visualStim_times'][:,0],
                                'response time':session['trials_response_times'][:,0],
                                'feedback time':session['trials_feedback_times'][:,0],
                                
                                #'intervals':session['trials_intervals'][:,0], # same as go cue
                                
                                
                                'go cue':session['trials_intervals'][:,0],
                                
                                'stim contrast left':session['trials_visualStim_contrastLeft'][:,0],
                                'stim contrast right':session['trials_visualStim_contrastRight'][:,0],
                                
                                'response choice':session['trials_response_choice'][:,0],
                                'response time':session['trials_response_times'][:,0],
                                'feedback type':session['trials_feedbackType'][:,0],
                                
                                })
        # add wheelmovement to trials dv ===============
        #wheel = W[np.newaxis, :, :]
        whl = pd.DataFrame( {'wheel movement':np.zeros(ntrials)}, dtype=object) 
        for row in range(W.shape[0]):
            whl['wheel movement'][row] = W[row,:]

        #merge trials with wheel
        trials_df = pd.merge(trials_df, whl, how='right', left_index=True, right_index=True)
    
        # add trials df to session
        session['trials_df'] = trials_df


        return session

# Helper Functions to Load Data ============================================================================
    def get_good_cells(self, folder):
        # location in brain of each neuron
        brain_loc = os.path.join(folder, "channels.brainLocation.tsv")

        good_cells = (np.load(os.path.join(folder, "clusters._phy_annotation.npy")) >= 2 ).flatten()
        clust_channel = np.load(os.path.join(folder, "clusters.peakChannel.npy")).astype(int) - 1
        br = []
        with open(brain_loc, 'r') as tsv:
            tsvin = csv.reader(tsv, delimiter="\t")
            k=0
            for row in tsvin:
                if k>0:
                    br.append(row[-1])
                k+=1
        br = np.array(br)
        good_cells = np.logical_and(good_cells, clust_channel.flatten()<len(br))
        brain_region = br[clust_channel[:,0]]


        return good_cells, brain_region, br


    def get_waves(self, folder):
        fname = os.path.join(folder, "clusters.waveformDuration.npy")
        twav = np.load(fname)

        fname = os.path.join(folder, "clusters.templateWaveforms.npy")
        W = np.load(fname)

        fname = os.path.join(folder, "clusters.templateWaveformChans.npy")
        ichan = np.load(fname).astype('int32')

        u = np.zeros((W.shape[0], 3, 384))
        w = np.zeros((W.shape[0], 82, 3))

        for j in range(W.shape[0]):
            wU  = TruncatedSVD(n_components = 3).fit(W[j]).components_
            wW = W[j] @ wU.T
            u[j][:, ichan[j]%384] = wU
            w[j] = wW

        return twav, w, u


    def get_probe(self, br, folder):
        prb_name = os.path.join(folder, "probes.rawFilename.tsv")
        prb = []
        with open(prb_name, 'r') as tsv:
            tsvin = csv.reader(tsv, delimiter="\t")
            for row in tsvin:
                prb.append(row[-1])
            prb = prb[1:]
        for ip in range(len(prb)):
            pparts = prb[ip].split('_')
            prb[ip] = '%s_%s_%s_%s'%(pparts[0], pparts[1], pparts[2], pparts[3])

        brow = []
        blfp = []
        for iprobe in range(len(prb)):
            ch_prb = np.load(os.path.join(folder, "channels.probe.npy")).astype(int)
            raw_row = np.load(os.path.join(folder, "channels.rawRow.npy")).astype(int)
            ich = (ch_prb.flatten()==iprobe).nonzero()[0]
            bunq = np.unique(br[ich])
            bunq = bunq[bunq!='root']
            nareas = len(bunq)
            brow.append([])
            for j in range(nareas):
                bid = br[ich]==bunq[j]
                brow[-1].append(raw_row[ich[bid], 0])
            blfp.append(bunq)
        return prb, blfp, brow


    def get_LFP(self, br, etime, dT, dt, T0, folder):
        prb, blfp, brow = get_probe(folder, br)
        bsize = 100000
        nbytesread = 385 * bsize * 2

        L = []
        BA_LFP = []
        for ip in range(len(prb)):
            BA_LFP.extend(blfp[ip])

            # root = self.folder
            # ''Z:/3 Projekte/Neuromatch Summer School/Projekt/steinmetz_dataset\\Cori_2016-12-14\\Cori_2016-12-14_M2_g0_t0.imec.lf.bin'
            # fname_lfp = '%s_t0.imec.lf.bin'%(prb[ip])

            # LFP = []
            # with open(os.path.join(root, fname_lfp), 'rb') as lfp_file:
            #     while True:
            #         buff = lfp_file.read(nbytesread)
            #         data = np.frombuffer(buff, dtype=np.int16, offset=0)
            #         if data.size==0:
            #             break
            #         data = np.reshape(data, (-1, 385))

            #         nareas = len(brow[ip])
            #         lfp = np.zeros((data.shape[0], nareas))
            #         for j in range(nareas):
            #             lfp[:,j] = data[:, brow[ip][j]].mean(-1)
            #         LFP.extend(lfp)
            # LFP = np.array(LFP)
            fname_lfp_times = '%s_t0.imec.lf.timestamps.npy'%(prb[ip])
            lfp_times = np.load(os.path.join(self.folder, fname_lfp_times))
            L.extend(ppsth(LFP, lfp_times,  etime, dT, dt))

        L = np.array(L)
        L = L - np.expand_dims(np.mean(L[:,:,:int(T0//dt)], axis=-1), axis=-1)

        return L, BA_LFP

    def get_passive(self, folder):
        vis_right_p = np.load(os.path.join(folder, "passiveVisual.contrastRight.npy")).flatten()
        vis_left_p = np.load(os.path.join(folder, "passiveVisual.contrastLeft.npy")).flatten()
        vis_times_p = np.load(os.path.join(folder,   "passiveVisual.times.npy"))
        return vis_times_p, vis_right_p, vis_left_p


    def get_event_types(self, folder):
        response = np.load(os.path.join(folder, "trials.response_choice.npy")).flatten()
        vis_right = np.load(os.path.join(folder, "trials.visualStim_contrastRight.npy")).flatten()
        vis_left = np.load(os.path.join(folder, "trials.visualStim_contrastLeft.npy")).flatten()
        feedback_type = np.load(os.path.join(folder, "trials.feedbackType.npy")).flatten()

        return response, vis_right, vis_left, feedback_type

    def get_event_times(self, folder):
        response_times = np.load(os.path.join(folder, "trials.response_times.npy"))
        visual_times = np.load(os.path.join(folder,   "trials.visualStim_times.npy"))
        gocue = np.load(os.path.join(folder,   "trials.goCue_times.npy"))
        feedback = np.load(os.path.join(folder,   "trials.feedback_times.npy"))

        rsp = response_times - visual_times
        feedback = feedback - visual_times
        gocue = gocue - visual_times

        return response_times, visual_times, rsp, gocue, feedback

    def get_wheel(self, folder):
        wheel = np.load(os.path.join(folder, "wheel.position.npy")).flatten()
        wheel_times = np.load(os.path.join(folder,   "wheel.timestamps.npy"))
        return wheel, wheel_times

    def get_pup(self, folder):
        pup = np.load(os.path.join(folder, "eye.area.npy"))
        pup_times = np.load(os.path.join(folder,  "eye.timestamps.npy"))
        xy = np.load(os.path.join(folder, "eye.xyPos.npy"))

        return pup, xy, pup_times

    def get_spikes(self, folder):
        stimes = np.load(os.path.join(folder, "spikes.times.npy")).flatten()
        sclust = np.load(os.path.join(folder, "spikes.clusters.npy")).flatten()
        return stimes, sclust

    def first_spikes(self, stimes, t0):
        tlow = 0
        thigh = len(stimes)

        while thigh>tlow+1:
            thalf = (thigh + tlow)//2
            sthalf = stimes[thalf]
            if t0 >= sthalf:
                tlow = thalf
            else:
                thigh = thalf
        return thigh

    def wpsth(self, wheel, wheel_times, etime, dT, dt):
        ntrials = len(etime)
        NT = int(dT/dt)
        f = interp1d(wheel_times[:,1], wheel_times[:,0], fill_value='extrapolate')
        S  = np.zeros((ntrials, NT))
        for j in range(ntrials):
            tsamp = f(np.arange(etime[j], etime[j]+dT+1e-5, dt)).astype('int32')
            S[j,:] = wheel[tsamp[1:]] - wheel[tsamp[:-1]]
        return S

    def ppsth(self, pup, pup_times, etime, dT, dt):
        nk = pup.shape[-1]
        ntrials = len(etime)
        NT = int(dT/dt)
        f = interp1d(pup_times[:,1], pup_times[:,0], fill_value='extrapolate')
        S  = np.zeros((nk, ntrials, NT))
        for k in range(nk):
            for j in range(ntrials):
                tsamp = f(np.arange(etime[j], etime[j]+dT-1e-5, dt) + dt/2).astype('int32')
                S[k, j,:] = pup[tsamp, k]
        return S

    def psth(self, stimes, sclust, etime, dT, dt):
        NN = np.max(sclust)+1
        NT = int(dT/dt)
        ntrials = len(etime)

        S  = np.zeros((NN, ntrials, NT))
        for j in range(ntrials):
            k1   = self.first_spikes(stimes, etime[j])
            k2   = self.first_spikes(stimes, etime[j]+dT)
            st   = stimes[k1:k2] - etime[j]
            clu  = sclust[k1:k2]
            S[:,j,:] = csr_matrix((np.ones(k2-k1), (clu, np.int32(st/dt))), shape=(NN,NT)).todense()

        return S

    def get_brain_region(self):
        regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain",  "basal ganglia", "subplate"]
        brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                        ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                        ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                        ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                        ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                        ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                        ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                    ]
        return regions, brain_groups
