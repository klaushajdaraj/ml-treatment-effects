# Casual Machine Learning for Heterogeneous Treatment Effects: An Empirical Application on Optimal Treatment Assignment

![Static Badge](https://img.shields.io/badge/python-3.9.13-blue?link=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-3913%2F)
![Static Badge](https://img.shields.io/badge/licence-MIT-red?link=https%3A%2F%2Fgithub.com%2Fklaushajdaraj%2Fml-treatment-effects%2Fblob%2Fmain%2FLICENSE)
![Static Badge](https://img.shields.io/badge/tensorflow-2.10.0-green?link=)

## Introduction

This repository contains the code, data, and documentation for my Master Thesis, titled Casual Machine Learning for Heterogeneous Treatment Effects: An Empirical Application on Optimal Treatment Assignment. The thesis explores the utilization of machine learning for improved causal inference. Included are all the necessary scripts and resources to reproduce the results, as well as detailed explanations of the methodologies used. Feel free to explore the materials and reach out if you have any questions or feedback!

## Main configurations:

Ran on:

- Windows 11
- Python 3.9.13
- tensoflow==2.10.0
- protobuf==3.11.3

## Files

### `repetitions_subsettreatments.joblib`

Contains the CV_Results (see mlmethods) saved from the hundred times performed three-folded cross validation Hitsch-Misra-Matching for two ML-Methods. Only treatments 1, 2, 4 and 5 were considered.

### `repetitions_alltreatments.joblib`

Contains the CV_Results (see mlmethods) saved from the hundred times performed three-folded cross validation Hitsch-Misra-Matching for two ML-Methods. All treatments were considered.

### `plots.py`

Code for creating plots used in the Analytics.ipynb which is the main Jupyter notebook for evaluating the results.

### `mlmethods.py`

Main script with two ML-Method classes and the code for Hitsch-Misra-Matching. It is only used for importing on the main script, empty `main()`.

### `expdata.csv`

Raw data of the experiment from Opitz et al. (2024).

### `cv_script.py`

Script for hyper-parameter tuning of the two ML-Methods.

### `Analytics.ipynb`

The main Jupyter notebook for creating descriptional statistics, result tables and figures.

### `misramatching_script.py`

Performs the Hitsch-Misra-Matching with the two ML methods. Adjust the `used_treatments` list for the subset of treatments. In addition, there can be found the dictionary with used hyperparameters.

Please note that the paths in the python scripts have to be adjusted to the user's working directory! Therefore, it is necessary to change the paths according to your local directories.
"# machine-learning-treatment-effects"
"# ml-treatment-effects"
