# NMA_project
## How to use loader object:
the new way to load a session is as follows:
import necessary libraries:

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    #import qgrid
    %matplotlib inline
    
    # import loader object
    from nma_class import loader


initialize loader 

    # link to parent folder with all datasets
    main_folder = "E:/3 Projekte/Neuromatch Summer School/Projekt/steinmetz_dataset"
    # initialize loader object
    loader = loader(main_folder)

then you have to get the available sessions of your folder like this:

    # get available sessions
    all_session_folders, all_session_names = loader.get_available_session()
    
and load the default session:

    # load default session = 
    session = loader.load_session(all_session_folders[0])
    spikes_df = session['spikes_df']
    clusters_df = session['clusters_df']
    trials_df = session['trials_df']
    
---

### alternatively you can also initizalize the loader object with the default session = first folder in main folder:

    # initialize with default session
    loader = loader(main_folder, init_default=True)
    session = loader.default_session

### to speed up loading process:
to spead up the loading process the `spikes_df`, `trials_df` and `clusters_df`can be exportet via pickl to 
a gzip file and later directly read without the need for parsing

read via fast method:

    load_session(folder, fast=True):

to write and or update files:

    load_session(folder, update=True):
    
# Plotting Functions:
collection of all plotting functions in nma_class 

import eda module and create object:

    from nma_class import eda

    data = [spikes_df, clusters_df, trials_df]
    params = dict()
    params['sampling_rate'] = 20000

    session_eda = eda(data, all_session_names[0], all_session_folders[0], params)

## plot spike trians
![](.images/spike_train.png?raw=true "example")

    cluster = 4
    spikes = clusters_df.loc[cluster, 'spikes']
    trials_select_df = trials_df.loc[trials_df['included']==True, ['start time','end time']]
    fig, ax = session_eda.plt_spike_train(cluster, spikes, trials_select_df)
    plt.show()

## plot spike trains around event with histogram
![](.images/spike_around_event.png?raw=true "example")

    title = None
    cluster = 8
    event = 'stim time'
    window = 1
    trials_select_df =  session['trials_df'].loc[session['trials_df']['included']==True, 'stim time']
    spikes = clusters_df.loc[8, 'spikes']

    fig, (ax1, ax2, ax3) = plt_spike_train_hist_bar(cluster, event, window, trials_select_df, spikes)
