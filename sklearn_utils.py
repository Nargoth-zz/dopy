##############################
### sklearn_utils: a small extension to the scikit-learn framework
### to produce plots and do calculations commonly used in HEP.
###
### Author: Timon Schmelzer (timon.schmelzer@tu-dortmund.de)
###
### Date (last update): 11.07.2016 (11.07.2016)
##############################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from tqdm import tqdm_notebook

# Plot functions:
def plot_roc_curve(clfs, X, y, labels=None, scale_xy=[[0.0, 1.0],[0.0, 1.0]]):
    """Plots a roc curve for one or multiple classifiers using array of features and corresponding flags.
    
    Keyword arguments:
        clfs -- single classifier or list of them
        X -- array of features
        y -- corresponding flags
        labels -- plot labels, '{}' will be replaced with the roc auc score (default: 'ROC curve (area = {:.4f})')
    """
    import collections
    
    if not isinstance(clfs, collections.MutableSequence):
        clfs = [clfs]
        
    if labels == None:
        labels = ['ROC curve (area = {:.4f})']*len(clfs)
        
    if not isinstance(labels, collections.MutableSequence):
        labels = [labels]
        
    # Check input data
    if len(labels) != len(clfs):
        raise ValueError('Number of classifier and labels have to be the same! {} vs. {}'.format(
            len(clfs), len(labels)))
    
    # Calculate roc curves, roc_auc and plot it
    print(len(clfs), len(labels))
    for i, (clf, label) in enumerate(zip(clfs, labels)):
        if len(X) != len(y):
            raise ValueError('Lenght of dataset and corresponding flags have to be the same! {} vs. {}'.format(
                len(X), len(y)))
        dec = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, dec)
        roc_auc = roc_auc_score(y, dec)        
        if i == 0:
            plt.plot([0, 1], [0, 1], 'k--', label='Random guessing')
            plt.xlim([scale_xy[0][0], scale_xy[0][1]])
            plt.ylim([scale_xy[1][0], scale_xy[1][1]])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, label=label.format(roc_auc))
        plt.legend(loc="lower right")

    plt.show()
    

def plot_classifier_output(clf, X_train, y_train, X_test=None, y_test=None, bins=50, title=None):
    """Plots classifier probability distributions from training (and testing) dataset.
    
    Keyword arguments:
        clf -- trained classifier
        X_train -- training dataset
        y_train -- corresponding training flags
        X_test -- testing dataset, optional (default: False)
        y_test -- corresponding testing flags, optional (default: False)
        bins -- number of bins (default: 50)
        title -- title of the plot (default: None)
    """    
    # Calculate probabilities
    probs_test = None
    try:
        probs_train = clf.decision_function(X_train)
#        if not X_test:
        probs_test = clf.decision_function(X_test)
    except:
        probs_train = clf.predict_proba(X_train)[:, 1]
#        if not X_test:
        probs_test = clf.predict_proba(X_test)[:, 1]

    # Plot training distribution
    _, binning, _ = plt.hist(np.array(probs_train[np.array(y_train) == 1]), alpha=0.5, label='SIG (Train)', 
                             color='b', normed=True, bins=bins)
    plt.hist(np.array(probs_train[np.array(y_train) == 0]), alpha=0.5, label='BKG (Train)', 
            color='r', normed=True, bins=binning)
    
    width = (binning[1] - binning[0])
    center = (binning[:-1] + binning[1:]) / 2

    # 1 = Signal, 0 = Background
#    if X_test != None and y_test != None:
    for i, label, color in zip(range(1, -1, -1), ['SIG (Test)', 'BKG (Test)'], ['b', 'r']):
        
        hist, _ = np.histogram(probs_test[np.array(y_test) == i],
                               bins=binning, normed=True)
        scale = len(probs_test[np.array(y_test) == i]) / sum(hist)
        err = np.sqrt(hist * scale) / scale
        
        plt.errorbar(center, hist, yerr=err, fmt='o', c=color, label=label)

    plt.legend(loc='best')
    plt.ylim(0)
    if title != None:
        plt.title(title)
    plt.show()
    
    
def plot_bdt_vars(df, flags, sig_label='Signal MC (Sig)', bkg_label='Data Upper SB (Bkg)', 
                  plot_appendix='', **kwargs):
    """Plots signal vs. background distributions.
    
    Keyword arguments:
        df -- pandas DataFrame containing all relevant observables
        flags -- corresponding flags, signal=1, background=0
        sig_label -- label of signal distribution (default: 'Signal MC (Sig)')
        bkg_label -- label of background distribution (default: 'Data Upper SB (Bkg)')
        plot_appendix -- appendix to the plot title (default: '')
        kwargs -- additional key word arguments for histogram plots
    """  
    import seaborn
    if len(df) != len(flags):
        raise ValueError('DataFrame of features and flags have to be of equally lenght! {} vs {}'.format(
            len(df), len(flags)))
    flags = np.array(flags)
    plots_in_x = 2
    plots_in_y = int(round(len(df.columns) / 2 + 0.4))
    
    plt.figure(figsize=(16, 8*plots_in_y))
    for i, var in tqdm_notebook(enumerate(df.columns, start=1), total=len(df.columns)):
        plt.subplot(plots_in_y, plots_in_x, i)
        _, binning, _ = plt.hist(df[var][flags == 1], bins=50, alpha=0.6, normed=True, 
                                 label=sig_label, **kwargs);
        plt.hist(df[var][flags == 0], bins=binning, alpha=0.6, normed=True, label=bkg_label,
                 **kwargs);
        plt.xlabel(var + plot_appendix)
        plt.legend(loc='best')
    plt.show()

def plot_feature_importances(clf, X):
    import seaborn
    importances_sorted = sorted(zip(clf.feature_importances_, X.columns), reverse=True)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), [val[0] for val in importances_sorted],
                color="r", alpha=0.5, align="center")
    plt.xticks(range(X.shape[1]),X.columns, rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()    
    
    
# Calculations:
def train_kfold(clf_type, X, y, folds=6, show_plots=False, write_decisions=False, state=0, **kwargs):
    """Uses kFolding to train a certain classifier.
    
    Keyword arguments:
        clf_type: classifier type as string, currently supported: 
                 ['AdaBoostClassifier', 'GradientBoostingClassifier']
        X: complete dataset (note: you don't need to split your dataset into a train and test 
           dataset using kFolding!)
        y: corresponding flags
        folds: number of folds (default: 6)
        show_plots: if True, shows probability distributions from training and testing dataset, 
                    using the 'plot_train_test_comparison' function (default: False)
        write_decisions: if True, appends decision columns to given DataFrame X (default: False)
        kwargs: key word arguments for KFold
        
    Returns:
        list of trained classifiers
    """
    if clf_type not in ['AdaBoostClassifier', 'GradientBoostingClassifier']:
        raise ValueError('Classifier type {} is not supported for kfolding right now!'.format(clf_type))

    decision_col_name = clf_type + '_decision'
    clfs = []
    
    kf = KFold(len(X), n_folds=folds, **kwargs)
    
    for i, (train_index, test_index) in tqdm_notebook(enumerate(kf, start=1), total=len(kf)):
        train_cols = list(X.columns)
        
        if write_decisions and decision_col_name in X.columns:
            train_cols.remove(decision_col_name)
                    
        X_train, X_test = X[train_cols].iloc[train_index], X[train_cols].iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if clf_type == 'AdaBoostClassifier':
            clf = AdaBoostClassifier(random_state=state)
        elif clf_type == 'GradientBoostingClassifier':
            clf = GradientBoostingClassifier(random_state=state)
            
        clf.fit(X_train.as_matrix(), y_train, sample_weight=res_scaled.iloc[train_index].sig_bkg_weights.as_matrix())
        
        if show_plots:
            plot_train_test_comparison(clf, X_train, y_train, X_test.as_matrix(), y_test.as_matrix(), 
                                       title='Classifier iteration {}'.format(i))

        if write_decisions:
            X.set_value(test_index, decision_col_name, clf.decision_function(X_test))
        
        clfs.append(clf)

    return clfs
