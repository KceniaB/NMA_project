import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import platform
import os, csv
from os import path
from numba import jit
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d
from sklearn.decomposition import TruncatedSVD


"""
Class for all the functions from Samuel, Kcenia and Max NMA Projekt


"""

class compute():
    """[class to compute stuff]

    Returns:
        [type]: [description]
    """    
    def __init__(self, session):
        self.session = session
        self.trials_df = session['trials_df']
        self.spikes_df = session['spikes_df']
        self.clusters_df = session['clusters_df']






# ===============================================================================================================================================



class loader():
    """[class to load each session and cleanup data]

    Returns:
        [type]: [description]
    """    
    def __init__(self, main_folder, init_default=False, dt=1/1000, dT=2.5, T0=0.5):
        """[initialize object]

        Args:
            #main_folder ([string]): [filepath to all the sessions folders]
            init_default (bool, optional): [initialize first folder in parent folder]. Defaults to True.
            dt ([type], optional): [timewindow for spikes_ar]. Defaults to 1/1000.
            dT (float, optional): [description]. Defaults to 2.5.
            T0 (float, optional): [start time for spikes_ar]. Defaults to 0.5.
        """        
        self.dt = dt
        self.dT = dT
        self.T0 = T0
        self.regions, self.brain_groups = self.get_brain_region()
        self.folder=main_folder
        self.subfolders, self.all_sessions = self.get_available_session()
        # load first session
        if init_default:
            self.default_session = self.load_session(self.subfolders[0])
    
    #  find all available sessions
    def get_available_session(self):
        subfolders = [ f.path for f in os.scandir(self.folder) if f.is_dir() ]
        sessions = [folder_.split('/')[-1] for folder_ in subfolders]
        return subfolders, sessions

    # load session from given sesion folder
    def load_session(self, folder, fast=False, update=False):
        """[load important information for each given session]

        Args:
            folder ([string]): [path to folder with session files to load]
            fast ([bool]): [true: complete dictionary isloaded, false: just trials_df, spikes_df, cluster_df, spikes_ar]
            update ([boo]): [true: trials_df, spikes_df, cluster_df, spikes_ar files are updated]

        Returns:
           session [dict]: [dictionary with keys:
                    folder: folder of session
                    spikes_df: dataframe with all spikes and cluster belonging to
                    clusters_df: infromation about neurons with spikes
                    trials_df: infromation about trials, times, and wheel movement
                    clusters = good_cells
                    brain_area = brain_regions
                    spikes_ar: binned spikes for each neuron and each trial
                    wheel: wheel movement per trial
                    pupil: pupil diameter per time bin
                    response = response
                    contrast_right: vis_right
                    contrast_left: vis_left
                    response_time: rsp
                    feedback_time: feedback
                    feedback_type: feedback_type  
                    gocue: gocue
                    mouse_name
                    date_exp
                    trough_to_peak
                    waveform_w
                    waveform_u
                    bin_size
                    stim_onset
                    spikes_ar_passive: binned spikes for each neuron and each passive trial
                    wheel_passive: wheel movement per passive trial
                    pupil_passive: 
                    contrast_right_passive: 
                    contrast_left_passive: ]
        """        
        # check if fast is selected

        files = ['trials_df.pd', 'spikes_df.pd', 'clusters_df.pd']#, 'spikes_ar.npy']
        files_exist = [ path.exists( os.path.join(folder, fi) ) for fi in files ]
        if fast and all(files_exist):
            session = dict()
            #session['trials_df'] = pd.read_csv(os.path.join(folder, 'trials_df.csv') )
            session['trials_df'] = pd.read_pickle(os.path.join(folder, 'trials_df.pd'), compression='gzip' )
            #session['spikes_df'] = pd.read_csv(os.path.join(folder, 'spikes_df.csv') )
            session['spikes_df'] = pd.read_pickle(os.path.join(folder, 'spikes_df.pd'), compression='gzip' )
            #session['clusters_df'] = pd.read_csv(os.path.join(folder, 'clusters_df.csv') )
            session['clusters_df'] = pd.read_pickle(os.path.join(folder, 'clusters_df.pd'), compression='gzip' )
            session['spikes_ar'] = np.load(os.path.join(folder, 'spikes_ar.npy') )
            #session['spikes_ar'] = sparse.load_npz(os.path.join(folder, 'spikes_ar.npy') ).toarray()



        else:
            # load session files into dictionary ===========================================

            ## load session files ===========================================
            #from os import listdir
            #from os.path import isfile, join
            #files = [f for f in listdir(folder) if isfile(join(folder, f))]
            #files = [f for f in files if f[:2]!='._'] #Samuel added this to deal with weird bug
            #names = ['_'.join(f.split('.')[:-1]) for f in files]
            #session = dict()
            #for (file_, name_) in zip(files, names):
            #    if file_.split('.')[-1] == 'npy':
            #        session[name_] = np.load(join(folder,file_))
            #    if file_.split('.')[-1] == 'tsv':
            #        session[name_] = pd.read_table(join(folder,file_))

            # Load Pachitariu Session Data ===========================================================================
            # good cells and brain regions
            good_cells, brain_region, br, phy_label = self.get_good_cells(folder)
            # get all spikes
            spk_spikes, spk_clusters = self.get_all_spikes(folder)
            # event types
            response, vis_right, vis_left, feedback_type, included_tr, rep_nrs = self.get_event_types(folder)
            # event timing
            response_times, visual_times, gocue_times, feedback_times, interval_times, rsp, feedback, gocue, interval = self.get_event_times(folder)   
            # get passive trials
            vis_times_p, vis_right_p, vis_left_p, gocue_p = self.get_passive(folder)
            visual_times = np.vstack((visual_times, vis_times_p))
            vis_right = np.hstack((vis_right, vis_right_p))
            vis_left  = np.hstack((vis_left, vis_left_p))
            # get spikes and clusters of spikes
            spikes, sclust    = self.get_spikes(folder)
            # wheel traces
            wheel, wheel_times = self.get_wheel(folder)
            # load the pupil
            pup, xy, pup_times = self.get_pup(folder)
            # load the LFP
            #L, ba_lfp = self.get_LFP(folder, br, visual_times-T0, dT, dt, T0)
            
            # wheel trials
            W = self.wpsth(wheel, wheel_times,   visual_times-self.T0, self.dT, self.dt)
            # pupil loader
            P = self.ppsth(pup, pup_times,   visual_times-self.T0, self.dT, self.dt)
            # add spike waveform information
            twav, w, u = self.get_waves(folder)
            
            
            # put all the variables into dictionary
            session = dict()
            # _p mens for passif trials
            # number of active trials
            ntrials = len(response)
            session['folder'] = folder
            session['clusters'] = np.arange(len(good_cells))[good_cells]
            session['brain_area'] = brain_region[good_cells]
            session['wheel'] = W[np.newaxis, :ntrials, :]
            session['pupil'] = P[:, :ntrials, :]
            session['response'] = response
            session['contrast_right'] = vis_right[:ntrials]
            session['contrast_left'] = vis_left[:ntrials]
            session['response_time'] = rsp
            session['feedback_time'] = feedback
            session['feedback_type'] = feedback_type  
            session['gocue'] = gocue
            #session['mouse_name'] = folder.split('\\')[1].split('_')[0]
            #session['date_exp'] = folder.split('\\')[1].split('_')[1]
            session['trough_to_peak'] = twav[good_cells].flatten()
            session['waveform_w'] = w[good_cells].astype('float32')
            session['waveform_u'] = u[good_cells].astype('float32')
            session['bin_size'] = self.dt
            session['stim_onset'] = self.T0
            session['wheel_passive'] = W[np.newaxis, ntrials:, :]
            session['pupil_passive'] = P[:, ntrials:, :]
            #session['lfp_passive'] = L[:, ntrials:, :]
            session['contrast_right_passive'] = vis_right[ntrials:]
            session['contrast_left_passive'] = vis_left[ntrials:]
            

            # pars probe infos ============================================================================
                # # load channel brain location infos
                # brain = pd.DataFrame(session['channels_brainLocation'])
                # # load channel probes
                # site = pd.DataFrame(session['channels_sitePositions'], columns=['channel 0', 'channel 1'])
                # # merge with channles_df
                # channels_df = pd.merge(brain, site, how='inner', left_index=True, right_index=True)
                # # load probe, row and site
                # meta = pd.DataFrame({'probe':session['channels_probe'][:,0],
                #                             'raw row':session['channels_rawRow'][:,0],
                #                             'channels_site':session['channels_site'][:,0] })
                # # merge metha with channel_df
                # channels_df = pd.merge(channels_df, meta, how='inner', left_index=True, right_index=True)
                # # add channels_df to session dictionary
                # session['channels_df']=channels_df

            # create spikes dataframe spikes_df ============================================================================
            # create spike_df tataframe, with each spike time and the cluster it belongs to
            spikes_df = pd.DataFrame( { 'cluster':spk_clusters[:,0], 'spike_times': spk_spikes[:,0] }) 
            # add spikes to session dictionary
            session['spikes_df']=spikes_df

            # create clusters/neurons dataframe clusters_df ============================================================================
            #clusters = np.unique(spikes_df['cluster'])
            # create number of spikes, phy2 manual cluster, 
            clusters = np.unique(spk_clusters)
            clusters_df = pd.DataFrame({'label':phy_label}, index=clusters )
            clusters_df['label'] = clusters_df['label'].apply( lambda label: 'good' if label==2 else ('mua' if label==1 else 'bad') )
            # drop rows with bad cells
            #clusters_df.drop( clusters_df[(clusters_df['label']=='bad')].index,axis=0,inplace=True)
            # create spikes colum with spiketimes

            spk = pd.DataFrame( {'spikes':np.zeros(len(clusters), dtype=object)}, index=clusters )
            # only select good clusters
            for group, frame in spikes_df.groupby('cluster'):
                spk['spikes'][group] = frame['spike_times'].values
            #merge spike column with clusters_df
            clusters_df = pd.merge(clusters_df, spk, how='right', left_index=True, right_index=True)
            # set index name
            clusters_df.index.name='cluster'
            # number of spikes per cluster
            clusters_df['total spikes'] = clusters_df['spikes'].apply(lambda row: len(row) )
            # add recording position
            clusters_df['recording area'] = brain_region

            session['clusters_df'] = clusters_df


            # create active trials dataframe activ_trials_df ============================================================================
            # create trials dataframe
            trials_df = pd.DataFrame({'included':included_tr,
                        'repetition number':rep_nrs,
                        'start time':interval_times[:,0],
                        'stim time':visual_times[:ntrials, :].flatten(),
                        #'go cue':gocue_times,
                        'response time':response_times.flatten(),
                        'feedback time':feedback_times.flatten(),
                        'end time':interval_times[:,1],
                        'stim contrast left':vis_left[:ntrials].flatten(),
                        'stim contrast right':vis_right[:ntrials].flatten(),
                        'response choice':response,
                        'feedback type':feedback_type,
                        })
            trials_df = trials_df.astype(dtype= {"repetition number":"int8", 
                                                'stim contrast left':'int8', 
                                                'stim contrast right':'int8',
                                                'response choice': 'int8',
                                                'feedback type':'int8'
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

        # write files if update or file does not exist
        if update #or any(files_exist):
            #session['trials_df'].to_csv(os.path.join(folder, 'trials_df.csv') )
            session['trials_df'].to_pickle(os.path.join(folder, 'trials_df.pd'), compression='gzip' )    
            #session['spikes_df'].to_csv(os.path.join(folder, 'spikes_df.csv') )
            session['spikes_df'].to_pickle(os.path.join(folder, 'spikes_df.pd'), compression='gzip' )
            #session['clusters_df'].to_csv(os.path.join(folder, 'clusters_df.csv') )
            session['clusters_df'].to_pickle(os.path.join(folder, 'clusters_df.pd'), compression='gzip' )
            # spikes are is too big - dont load it by default
            #np.save( os.path.join(folder, 'spikes_ar.npy'), session['spikes_ar'], allow_pickle=True )
            #sparse.save_npz(os.path.join(folder, 'spikes_ar.npy'), sparse.csr_matrix(session['spikes_ar']) )
        

        return session
    
    # load binned spikes for all trials and all sessions
    def binned_spikes(self, folder, TO=None, dT=None, dt=None):
        """[load spikes for all good neurons an all trials binned to 250 bins per trial]

        Args:
            folder ([string]): [folder of session to load, default =]
            T0 ([float]): [start time for each trial bin default T0=0.5]
            dT ([float]): [bin width, default dT=2.5]
            dt ([float]): [bin steps default  dt=1/1000]

        Returns:
            spikes_ar ([numpy array]): [spikes of active trials (neuron, trial, bin)]
            spikes_ar ([numpy array]): [spikes of passive trials (neuron, trial_passiv, bin)]
        """        
        # ititialize default values:
        if T0 is None:
            TO=self.T0
        if dT is None:
            dT=self.dT
        if dt is None:
            dt=self.dt
        # event times
        response_times, visual_times, _, _, _, _, _, _,  = self.get_event_times(folder)   
        # get spikes and clusters of spikes
        spikes, sclust = self.get_spikes(folder)
        # trials loader
        ntrials = len(response_times)
        S  = self.psth(spikes, sclust,   visual_times-self.T0, self.dT, self.dt)
        S  = S[good_cells].astype('int8') 
        spikes_ar=S[:, :ntrials, :]
        spikes_ar_passive = S[:, ntrials:, :]
        return spikes_ar, spikes_ar_passive

    # save wheel movement per trial to numpy array in session folder
    def save_wheel_to_npy(self, session):
        """[save the wheel movement of active and passive trials to npy file]

        Args:
            session ([dict]): [session to save wheel movement from]
        """        
        folder = session['folder']
        # save wheel movement of active trials
        fname = os.path.join(folder, "wheel_per_trial.npy")
        numpy.save(fname, session['wheel'], allow_pickle=True)
        # save wheel movement of passive trials
        fname = os.path.join(folder, "wheel_per_trial_p.npy")
        numpy.save(fname, session['wheel_passive'], allow_pickle=True)
        

    # get spikes for given array within start stop sequence
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

 # Helper Functions to Load Data ============================================================================
    def get_good_cells(self, folder):
        # location in brain of each neuron
        brain_loc = os.path.join(folder, "channels.brainLocation.tsv")
        phy_label = np.load(os.path.join(folder, "clusters._phy_annotation.npy")).flatten()
        phy_label_good = (phy_label >= 2)
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
        good_cells = np.logical_and(phy_label_good, clust_channel.flatten()<len(br))
        brain_region = br[clust_channel[:,0]]


        return good_cells, brain_region, br, phy_label

    def get_all_spikes(self, folder):
        fname = os.path.join(folder, "spikes.times.npy")
        spikes = np.load(fname)
        fname = os.path.join(folder, "spikes.clusters.npy")
        clusters = np.load(fname)

        return spikes, clusters

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
            lfp_times = np.load(os.path.join(folder, fname_lfp_times))
            L.extend(ppsth(LFP, lfp_times,  etime, dT, dt))

        L = np.array(L)
        L = L - np.expand_dims(np.mean(L[:,:,:int(T0//dt)], axis=-1), axis=-1)

        return L, BA_LFP

    def get_passive(self, folder):
        vis_right_p = np.load(os.path.join(folder, "passiveVisual.contrastRight.npy")).flatten()
        vis_left_p = np.load(os.path.join(folder, "passiveVisual.contrastLeft.npy")).flatten()
        vis_times_p = np.load(os.path.join(folder,   "passiveVisual.times.npy"))
        gocue_p = np.load(os.path.join(folder, "passiveBeeps.times.npy"))

        return vis_times_p, vis_right_p, vis_left_p, gocue_p


    def get_event_types(self, folder):
        response = np.load(os.path.join(folder, "trials.response_choice.npy")).flatten()
        vis_right = np.load(os.path.join(folder, "trials.visualStim_contrastRight.npy")).flatten()
        vis_left = np.load(os.path.join(folder, "trials.visualStim_contrastLeft.npy")).flatten()
        feedback_type = np.load(os.path.join(folder, "trials.feedbackType.npy")).flatten()
        included_tr= np.load(os.path.join(folder, "trials.included.npy")).flatten()
        rep_nr= np.load(os.path.join(folder, "trials.repNum.npy")).flatten()

        return response, vis_right, vis_left, feedback_type, included_tr, rep_nr

    def get_event_times(self, folder):
        response_times = np.load(os.path.join(folder, "trials.response_times.npy"))
        visual_times = np.load(os.path.join(folder,   "trials.visualStim_times.npy"))
        gocue_times = np.load(os.path.join(folder,   "trials.goCue_times.npy"))
        feedback_times = np.load(os.path.join(folder,   "trials.feedback_times.npy"))
        interval_times = np.load(os.path.join(folder, "trials.intervals.npy"))

        rsp = response_times - visual_times
        feedback = feedback_times - visual_times
        gocue = gocue_times - visual_times
        interval = interval_times - visual_times

        return response_times, visual_times, gocue_times, feedback_times, interval_times, rsp, feedback, gocue, interval

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


