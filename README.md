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




