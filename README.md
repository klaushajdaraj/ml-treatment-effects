# Main configurations:

Ran on:
- Windows 11
- Python 3.9.13
- tensoflow==2.10.0
- protobuf==3.11.3

# Files

### `repetitions_subsettreatments.joblib`

Contains the CV_Results (see mlmethods) saved from the hundred times performed three-folded cross validation Hitsch-Misra-Matching for two ML-Methods + combinations with all shrinkage methods. Only treatments 1, 2, 4 and 5 were considered.

### `repetitions_alltreatments.joblib`

Contains the CV_Results (see mlmethods) saved from the hundred times performed three-folded cross validation Hitsch-Misra-Matching for two ML-Methods + combinations with all shrinkage methods. All treatments were considered.

### `plots.py`

Code for creating plots used in the Analytics.ipynb.

### `mlmethods.py`

Main script with two ML-Method classes and the code for Hitsch-Misra-Matching. Is only used for importing, empty `main()`.

### `expdata.csv`

Raw data of the experiment.

### `cv_script.py`

Script for hyper-parameter tuning of the two ML-Methods.

### `Analytics.ipynb`

Jupyter notebook for creating descriptional statistics, result tables and figures.

### `misramatching_script.py`

Performs the Hitsch-Misra-Matching with the two ML methods and if wished with applied shrinkers. Adjust the `used_treatments` list for the subset of treatments. Here also the dictionary for the used hyperparameters can be found.

Please note that the paths in the python scripts have to be adjusted to your working directory!
"# machine-learning-treatment-effects" 
"# ml-treatment-effects" 
