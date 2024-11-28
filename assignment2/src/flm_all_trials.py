import os 
import numpy as np
import pickle
import pandas as pd
import nilearn
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel


def beta_check(filepath):
    if os.path.exists(filepath):
        try:
            f = open(filepath, 'rb')
            models_t, lsa_dm, b_maps, conditions_labels = pickle.load(f)
            return 'four_vars'
        except ValueError: 
            f = open(filepath, 'rb')
            models_t, lsa_dm = pickle.load(f)
            return 'two_vars'
    else: 
        return 'zero_vars'
        f.close() 

def def_design_matrix(sub_ind, models_events, models_confounds):
    """ Extracting design matrices for a single participant. One column per trial. sub_ind
     is the index of the participant used """
    lsa_dm=[]  # lsa_dm = least squares all design matrix
    for ii in range(len(models_events[sub_ind])):
        N=models_events[sub_ind][ii].shape[0]
        t_fmri = np.linspace(0, 600,600,endpoint=False) 
        trials = pd.DataFrame(models_events[sub_ind][ii], columns=['onset'])
        trials.loc[:, 'duration'] = 2 
        trials.loc[:, 'trial_type'] = [models_events[sub_ind][ii]['trial_type'][i-1]+'_'+'t_'+str(i).zfill(3)  for i in range(1, N+1)]

        lsa_dm.append(make_first_level_design_matrix(
            frame_times=t_fmri, 
            events=trials,
            add_regs=models_confounds[sub_ind][ii], 
            hrf_model='glover',
            drift_model='cosine'))
    return lsa_dm

def fit_model(sub_ind, lsa_dm, models, models_events, models_run_imgs):
    model_t = []
    for ii in range(len(models_events[sub_ind])):
        imgs = models_run_imgs[sub_ind][ii]
        model_t.append(FirstLevelModel())
        print(f'Fitting GLM no. {ii+1} for sub:', models[sub_ind].subject_label)
        model_t[ii].fit(run_imgs = imgs, design_matrices = lsa_dm[sub_ind][ii])
    return model_t


def flm_tbt_all_subs(flm_path, tbt_flm_path):
    f = open(flm_path, 'rb')
    models, models_run_imgs, models_events, models_confounds = pickle.load(f)
    f.close()

    lsa_dm = []
    for i in range(len(models)):
        lsa_dm.append(def_design_matrix(i, models_events, models_confounds))
    print("All design mattrices defined")
    models_t = []
    for i in range(len(models)):
        models_t.append(fit_model(i, lsa_dm, models, models_events, models_run_imgs))
    
    f = open(tbt_flm_path, 'wb')
    pickle.dump([models_t, lsa_dm], f)
    f.close()
    print("models and lsa_dms saved to out folder")
    return models_t, lsa_dm


def beta_maps(sub_ind, models_events, lsa_dm, models_t):
    b_map = []
    conditions_label = []

    for ii in range(len(models_events[sub_ind])):
        N = models_events[sub_ind][ii].shape[0]
        contrasts = np.eye(N) 
        dif = lsa_dm[sub_ind][ii].shape[1]-contrasts.shape[1] 
        contrasts = np.pad(contrasts, ((0,0),(0,dif)),'constant')    
        print(f'Making {N} contrasts for run : {ii+1}')
        for i in range(N):
            b_map.append(models_t[sub_ind][ii].compute_contrast(contrasts[i,], output_type='effect_size')) 
            conditions_label.append(lsa_dm[sub_ind][ii].columns[i]) 
    return b_map, conditions_label

def beta_maps_all(flm_path, tbt_flm_path):
    f = open(flm_path, 'rb')
    models, models_run_imgs, models_events, models_confounds = pickle.load(f)
    f.close()

    f = open(tbt_flm_path, 'rb')
    models_t, lsa_dm = pickle.load(f)
    f.close

    b_maps = []
    conditions_labels = []
    for i in range(len(models)):
        print(f'Making beta maps for sub-{models[i].subject_label}')
        b_map, conditions_label = beta_maps(i, models_events, lsa_dm, models_t)
        b_maps.append(b_map)
        conditions_labels.append(conditions_label)
    
    f = open(tbt_flm_path, 'wb')
    pickle.dump([models_t, lsa_dm, b_maps, conditions_labels], f)
    f.close()
    print("models lsa_dms, beta maps and condition labels saved to out folder")


def main():
    flm_path = "out/all_flm_IGT.pkl"
    tbt_flm_path = 'out/tbt_all_flm_IGT.pkl'

    if beta_check(tbt_flm_path) == 'zero_vars':
        print("Fitting models and creating beta maps...")
        models_t, lsa_dm,  = flm_tbt_all_subs(flm_path, tbt_flm_path)
        beta_maps_all(flm_path, tbt_flm_path)
    elif beta_check(tbt_flm_path) == 'two_vars':
        print("Creating beta maps from fitted models...")
        beta_maps_all(flm_path, tbt_flm_path)
    elif beta_check(tbt_flm_path) == 'four_vars':
        print('Fitted models, lsa_dm matrices, beta maps and condition labels have all been saved to the out folder')

if __name__ == "__main__":
    main()