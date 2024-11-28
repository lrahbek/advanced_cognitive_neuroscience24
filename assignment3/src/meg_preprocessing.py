import os 
import glob
import mne
from mne.io import concatenate_raws, read_raw_fif
import matplotlib.pyplot as plt
import numpy as np


def load_MEGdata(ID, data_folder):
    """ Load raw data from MEG_data folder for a given participant """
    paths = glob.glob(os.path.join(f"{data_folder}/{ID}/*_000000/MEG/*/files/*.fif"))
    raw_ls = []
    for i in range(len(paths)):
        if (paths[i].split("/")[-1] == "sessA.fif" or paths[i].split("/")[-1] == "sessB.fif"):
            raw_ls.append(mne.io.read_raw_fif(paths[i], preload = True))
    raw = concatenate_raws(raw_ls, on_mismatch = "ignore")
    return raw

def LP_filt(raw, low_pass):
    """ apply low pass filter to raw file"""
    raw = raw.filter(l_freq=None, h_freq=low_pass)
    return raw

def fit_plot_ICA(raw, out_path):
    """ Fit ICA to raw data and save plots of components to given outpath """
    ica = mne.preprocessing.ICA(n_components = 10, random_state = 42, max_iter = 800)
    ica.fit(raw)
    ica.exclude = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    ica_plots = ica.plot_properties(raw, picks = ica.exclude, show = False)
    for i in range(len(ica_plots)):
        out_path_i = f"{out_path}{i}"
        ica_plots[i].savefig(out_path_i)
    return ica

def apply_ica(raw, ica, excl):
    """ Apply fitted ica to raw data, and exclude given components return orig_raw and new raw"""
    ica.exclude = excl
    orig_raw = raw.copy()
    raw.load_data()
    ica.apply(raw)
    return raw, orig_raw

def extract_events(raw, class1_trig, class2_trig):
    """ Extract relevant events from data and collapse relevant events into binary ones """ 
    events = mne.find_events(raw, min_duration=0.002, consecutive=True)
    subset_ind = np.argwhere(np.isin(events[:, 2], class1_trig+class2_trig)).ravel()
    subset_events = mne.merge_events(events, class1_trig, 100, replace_events=True)
    subset_events = mne.merge_events(subset_events, class2_trig, 200, replace_events=True)
    return subset_events

def def_epochs(raw, subset_events, event_id_bin, classes):
    """ Define epochs from relevant events """ 
    epochs = mne.Epochs(raw, subset_events, event_id_bin, on_missing='warn', 
                        tmin=-0.200, tmax=1.000, baseline=(None, 0), preload = True, proj = False)
    epochs.pick_types(meg=True, eog=False, ias=False, emg=False, misc=False, stim=False, syst=False)
    epochs.equalize_event_counts(classes, random_state=42)
    return epochs