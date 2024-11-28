import os 
import glob
import mne
import pickle
from mne.io import concatenate_raws, read_raw_fif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, permutation_test_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from meg_preprocessing import load_MEGdata, LP_filt, fit_plot_ICA, apply_ica, extract_events, def_epochs

def preprocessing():
    """ Preprocess data """ 
    ID = "0145"
    data_folder = "/work/MEG_data"
    raw = load_MEGdata(ID, data_folder)
    print(f"Raw data loaded for {ID}")
    raw = LP_filt(raw, low_pass = 40)
    print("Low pass filter of 40Hz applied")
    out_path_ica = "../out/ICA/ICA00"
    ica = fit_plot_ICA(raw, out_path_ica)
    print("ica fitted and plots saved")
    exclusion_components = [0, 1, 2, 6, 7, 8, 9] #decided by visual inspection
    raw, _ = apply_ica(raw, ica, excl = exclusion_components) 
    win_trigs = [210, 220, 230, 240]
    loss_trigs = [211, 221, 231, 241]
    subset_events = extract_events(raw, class1_trig = win_trigs, class2_trig = loss_trigs)
    print("Events extracted")
    event_id_bin = {"Win":100, "Loss": 200}
    classes = ["Win", "Loss"]
    epochs = def_epochs(raw, subset_events, event_id_bin, classes) 
    print("Epochs defined")
    return raw, epochs

def split_prep_data(epochs, raw):
    """ Scale and transform data, and the split into test and train sets"""
    X_e = epochs.get_data(copy = True)
    y = epochs.events[:,2]
    X = mne.decoding.Scaler(raw.info).fit_transform(X_e, y)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    return X_train, X_test, y_train, y_test, X, y

def fit_eval_GNB(X_train, X_test, y_train, y_test, labels, report_path, confmatrix_path):
    """ Fit gaussian naive bayes to training data and evaluate on test data"""
    GNB = GaussianNB().fit(X_train, y_train)
    y_pred_GNB = GNB.predict(X_test)
    GNB_rep = classification_report(y_test, y_pred_GNB, target_names = labels)
    report_path_w = open(f"{report_path}GNB", "w")
    report_path_w.write(GNB_rep)
    report_path_w.close()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_GNB), display_labels = labels).plot()
    plt.savefig(f"{confmatrix_path}GNB.png")
    print("GNB fitted")

def fit_eval_MLP(X_train, X_test, y_train, y_test, labels, report_path, confmatrix_path):
    """ Fit MLP neural network to training data and evaluate on test data"""
    MLP = MLPClassifier(early_stopping = True, random_state=42, verbose = False).fit(X_train, y_train)
    y_pred_MLP = MLP.predict(X_test)
    MLP_rep = classification_report(y_test, y_pred_MLP, target_names = labels)
    report_path_w = open(f"{report_path}MLP", "w")
    report_path_w.write(MLP_rep)
    report_path_w.close()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_MLP), display_labels = labels).plot()
    plt.savefig(f"{confmatrix_path}MLP.png")
    plt.figure(figsize = (12,6))
    plt.subplot(1,2,1)
    plt.plot(MLP.loss_curve_)
    plt.title("Loss curve for MLP classifier")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.subplot(1,2,2)
    plt.plot(MLP.validation_scores_)
    plt.title("Validation scores")
    plt.xlabel("Iterations")
    plt.ylabel("accuracy")
    plt.savefig("../out/results/losscurve_MLP.png")
    print("MLP fitted")

def fit_eval_LRC(X_train, X_test, y_train, y_test, labels, report_path, confmatrix_path):
    """ Fit Logistic Regression classifier to training data and evaluate on test data"""
    LRC = LogisticRegression(random_state=42).fit(X_train, y_train)
    y_pred_LRC = LRC.predict(X_test)
    LRC_rep = classification_report(y_test, y_pred_LRC, target_names = labels)
    report_path_w = open(f"{report_path}LRC", "w")
    report_path_w.write(LRC_rep)
    report_path_w.close()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_LRC), display_labels = labels).plot()
    plt.savefig(f"{confmatrix_path}LRC.png")
    print("LRC fitted")

def permutations(estimator, X, y, permu_path, estimator_name):
    """ Run permutationtest with 100 permutation and cv = 5 using a given estimator """
    score, permutation_scores, pvalue = permutation_test_score(
        estimator = estimator, 
        X = X, y = y, 
        cv = 5, n_permutations = 100, 
        n_jobs = -1, 
        random_state = 42, verbose = 10)
    return [estimator_name, score, permutation_scores, pvalue]

def permutations_multiple_plot(permu_path, X, y):
    """ Run permutation test on GNB, MLP and LRC and save and plot results """
    permutation_eval = pd.DataFrame(columns = ["estimator", "score", "permutation_scores", "pvalue"])
    estimators = [GaussianNB(), 
                  MLPClassifier(early_stopping = True, random_state=42),
                  LogisticRegression(random_state=42)]
    est_names = ["GNB", "MLP", "LRC"]
    for i in range(len(estimators)): 
        permutation_eval.loc[len(permutation_eval)] = permutations(estimators[i], X, y, permu_path, est_names[i])
        print(f"Permutations for {est_names[i]} have finished")
    permutation_eval.to_csv(f"{permu_path}scores.csv", columns = ["estimator", "score", "pvalue"])
    plt.figure()
    colors = ["blue", "green", "yellow"]
    line_colors = ["--b", "--g", "--y"]
    ylim = (0, 20)
    for i in range(len(permutation_eval)):
        est = permutation_eval["estimator"].loc[i] 
        pval = permutation_eval["pvalue"].loc[i] 
        score = permutation_eval["score"].loc[i] 
        plt.hist(permutation_eval["permutation_scores"].loc[i], 20,
                edgecolor='black', linewidth = 1, color = colors[i], alpha = 0.6)
        plt.plot(2 * [score], ylim, line_colors[i], linewidth=3, label= f'Score [{est}] (pvalue: {pval})')
        plt.legend()
    plt.plot(2 * [1. / 2], ylim, '--k', linewidth=3, label='Chance level')
    plt.xlabel('Score')
    plt.savefig("../out/results/permutation_plot.png")
    print("permutations finished")


def main():
    labels = ["Win", "Loss"]
    raw, epochs = preprocessing()
    X_train, X_test, y_train, y_test, X, y = split_prep_data(epochs, raw)
    report_path = "../out/results/metrics_"
    confmatrix_path = "../out/results/confmatrix_"
    fit_eval_GNB(X_train, X_test, y_train, y_test, labels, report_path, confmatrix_path)
    fit_eval_MLP(X_train, X_test, y_train, y_test, labels, report_path, confmatrix_path)
    fit_eval_LRC(X_train, X_test, y_train, y_test, labels, report_path, confmatrix_path)
    del X_train, X_test, y_train, y_test
    permu_path = "../out/results/permu_"
    permutations_multiple_plot(permu_path, X, y)

if __name__ == "__main__":
    main()
