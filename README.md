# `sklearn_utils`: a small extension to the scikit-learn framework to produce plots and do calculations commonly used in HEP.

### Usage

Clone this repository to your favourite folder and import the `sklearn_utils.py` file to your Python file.

```
from sklearn_utils import train_kfold, plot_roc_curve

trained_clfs = train_kfold('AdaBoostClassifier', myX, myy, folds=10, shuffle=True)  # Uses kFolding to train multiple classifier

plot_roc_curve(trained_clfs, myX, myy)  # Plots ROC curves
```

Warning: Most (plot-)functions in `sklearn_utils` are written for iPython Notebooks. If you want to use _real_ python, you may apply some adjustments to make it work!

### Tips

If your folder structure looks like this

|-sklearn_utils

|---sklearn_utils.py

|---README.md

|-work_folder

|---myscript.py

and you want to import `sklearn_utils` in your `myscript.py`, try this:

```
# Other imports here...

import sys
sys.path.append('..')
from sklearn_utils.sklearn_utils import plot_roc_curve
```

### The package contains the following functions

* `plot_roc_curve(clfs, Xs, ys, labels=None, save=False, scale_xy=[[0.0, 1.0],[0.0, 1.0]], savepath_base='')`
* `plot_classifier_output(clf, X_train, y_train, X_test=None, y_test=None, bins=50, title=None, save=False, savepath_base='')`
* `plot_bdt_vars(df, flags, sig_label='Signal MC (Sig)', bkg_label='Data Upper SB (Bkg)', sig_name='SigMC', bkg_name='DataUpperSB', plot_appendix='', save=False, savepath_base='', **kwargs)` 
* `plot_feature_importances(clf, X, save=False, savepath_base='')`
* `train_kfold(clf_type, X, y, folds=6, show_plots=False, write_decisions=False, state=0, **kwargs)`
* `plot_correlations(data, save=False, savepath="", **kwds)`





