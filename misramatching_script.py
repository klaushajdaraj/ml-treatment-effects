# -*- coding: utf-8 -*-
"""
@author: Klaus Hajdaraj
"""

import pandas as pd
import yaml
import joblib

from mlmethods import *

# Read file paths
with open('config.yaml', 'r') as file:
    file_path = yaml.safe_load(file)

# Read the path to data and Documents (in my computer)
data_file_path = file_path["paths"]["data"]
documents_file_path = file_path["paths"]["documents"]

# Set path to results for full set of treatments (all six), or the subset of treatments (1, 2, 4, 5)
results_full_set_path = documents_file_path + "repetitions_alltreatments.joblib"
results_sub_set_path = documents_file_path + "repetitions_subsettreatments.joblib"

END_OF_X = 48
START_OF_X = 2
TREAT_NAME = "treat"
Y_NAME = "buttonpresses"
CONTROL_INDEX = 7
PATH = data_file_path

# Parameters used in the Final Model
params_allmodels = {
        "CausalNet": {
            "treat_1": {
                "params": {
                    "optimizer":"GradientDescent",
                    "learning_rate":0.01,
                    "alpha":0.1,
                    "r_par":0.6,
                    "hidden_layer_sizes":[20],
                    "dropout_rates":[0.5],
                    "batch_size":None,
                    "max_epochs_without_change":60,
                    "max_nepochs":10000,
                    "seed":42,
                    "verbose":False},"score":-8597310661459970.0},
            "treat_2": {
                "params":{"optimizer":"GradientDescent",
                          "learning_rate":0.01,
                          "alpha":0.01,
                          "r_par":1.0,
                          "hidden_layer_sizes":[20],
                          "dropout_rates":[0.5],
                          "batch_size":None,
                          "max_epochs_without_change":60,
                          "max_nepochs":10000,
                          "seed":42,
                          "verbose":False},"score":-13484863488.000002},
            "treat_3": {
                "params":{
                    "optimizer":"GradientDescent",
                    "learning_rate":0.01,
                    "alpha":0.1,
                    "r_par":1.0,
                    "hidden_layer_sizes":[20],
                    "dropout_rates":[0.5],
                    "batch_size":None,
                    "max_epochs_without_change":60,
                    "max_nepochs":10000,
                    "seed":42,
                    "verbose":False},"score":-94330435928063.95},
            "treat_4":{
                "params":{
                    "optimizer":"GradientDescent",
                    "learning_rate":0.01,
                    "alpha":0.01,
                    "r_par":0.0,
                    "hidden_layer_sizes":[34],
                    "dropout_rates":[0.5],
                    "batch_size":None,
                    "max_epochs_without_change":60,
                    "max_nepochs":10000,
                    "seed":42,
                    "verbose":False},"score":-583906340372479.6},
            "treat_5":{
                "params":{
                    "optimizer":"GradientDescent",
                    "learning_rate":0.01,
                    "alpha":0.1,
                    "r_par":0.6,
                    "hidden_layer_sizes":[100],
                    "dropout_rates":[0.5],
                    "batch_size":None,
                    "max_epochs_without_change":60,
                    "max_nepochs":10000,
                    "seed":42,
                    "verbose":False},"score":-565716675723265.4},
            "treat_6":{
                "params":{
                    "optimizer":"GradientDescent",
                    "learning_rate":0.01,
                    "alpha":0.01,
                    "r_par":0.0,
                    "hidden_layer_sizes":[60],
                    "dropout_rates":[0.5],
                    "batch_size":None,
                    "max_epochs_without_change":60,
                    "max_nepochs":10000,
                    "seed":42,
                    "verbose":False},"score":-444754936463360.1}
                    },

        "CausalForest": {
            'treat_1': {
                'max_features': 0.2,
                'max_samples': 0.1,
                'min_samples_leaf': 20,
                'min_var_fraction_leaf': 0.2,
                'max_depth': 25},
            'treat_2': {
                'max_features': 0.2,
                'max_samples': 0.1,
                'min_samples_leaf': 50,
                'min_var_fraction_leaf': None,
                'max_depth': 75},
            'treat_3': {
                'max_features': 0.8,
                'max_samples': 0.2,
                'min_samples_leaf': 5,
                'min_var_fraction_leaf': 0.3,
                'max_depth': 25},
            'treat_4': {
                'max_features': 0.2,
                'max_samples': 0.1,
                'min_samples_leaf': 50,
                'min_var_fraction_leaf': None,
                'max_depth': 5},
            'treat_5': {
                'max_features': 0.4,
                'max_samples': 0.2,
                'min_samples_leaf': 5,
                'min_var_fraction_leaf': None,
                'max_depth': 50},
            'treat_6': {
                'max_features': 0.3,
                'max_samples': 0.1,
                'min_samples_leaf': 10,
                'min_var_fraction_leaf': 0.2,
                'max_depth': None}
                }                    
        }

def getMinMax(df_list):
    min_df = df_list[0].copy()
    max_df = df_list[0].copy()
    for rowIndex, row in min_df.iterrows():
        for columnIndex, vlaue in row.items():
            min_df.loc[rowIndex,columnIndex] = min(
                [df.loc[rowIndex,columnIndex] for df in df_list])
            max_df.loc[rowIndex,columnIndex] = max(
                [df.loc[rowIndex,columnIndex] for df in df_list])
    return (min_df, max_df)

def getTreatDiff(stats):
    return round(stats.iloc[:-3,-1][::2].max() - stats.iloc[-2].max())

df = pd.read_csv(PATH)
used_treatments = ['treat_1', 'treat_2', 'treat_3', 'treat_4', 'treat_5', 'treat_6'] # Remove 'treat_3' and 'treat_6' when running subtreatments analysis
Matcher = MisraMatching(params_allmodels)
CV_Results = Matcher.cross_validate(
    data=df,
    repetitions=100,
    folds=3,
    treatment_names=used_treatments,
    used_estimators=[
        CausalNets,
        CausalForestHTE
    ],
)


min_df, max_df = getMinMax(CV_Results.rep_stats)
    
# Save the Excel tables with results
# writer = pd.ExcelWriter(results_full_set_path, engine='xlsxwriter') # Change directory here according to subset of treatments
writer = pd.ExcelWriter(results_sub_set_path, engine='xlsxwriter')
# writer = pd.ExcelWriter(documents_file_path + "overfit.xlsx", engine='xlsxwriter')

CV_Results.overall_stats.to_excel(writer, sheet_name='Overall')
min_df.to_excel(writer, sheet_name='Minimum')
max_df.to_excel(writer, sheet_name='Maximum')

writer.close()

# Save the results in the "joblib" format
joblib.dump(CV_Results, results_full_set_path) # Change directory here according to subset of treatments
joblib.dump(CV_Results, results_sub_set_path)
joblib.dump(CV_Results, documents_file_path + "overfit.joblib")