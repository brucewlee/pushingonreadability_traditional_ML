'''
    Module for preparing data for regression and classification.
'''
import pandas as pd
import numpy as np
import math
import tqdm
import csv
from typing import Dict, List, Tuple, Callable
from sklearn import preprocessing

def normalize(df, active_features):
    result = df.copy()
    for feature_name in df.columns:
        if active_features.count(feature_name) == 1:
            max_value = float(df[feature_name].max())
            min_value = float(df[feature_name].min())
            if max_value - min_value != 0:
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            else:
                result[feature_name] = df[feature_name] - min_value
    return result

def prepare_classification_data(df: pd.DataFrame,neural_models_to_use,this_handcrafted_features_list):
    # Features List
    # FeatureSet_Total_HF +
    # soft os : ["bert.14.prob.1","bert.14.prob.2","bert.14.prob.3"]
    # soft wb : ["bert.14.prob.1","bert.14.prob.2","bert.14.prob.3","bert.14.prob.4","bert.14.prob.5"]
    # soft cb : ["bert.14.prob.1","bert.14.prob.2","bert.14.prob.3","bert.14.prob.4","bert.14.prob.5"]

    if neural_models_to_use == "none":
        active_features = this_handcrafted_features_list
    else:
        this_neural_model_list = []
        '''
        for neural_model in neural_models_to_use:
            this_neural_model_list.append(neural_model+".14.prob.1")
            this_neural_model_list.append(neural_model+".14.prob.2")
            this_neural_model_list.append(neural_model+".14.prob.3")
            this_neural_model_list.append(neural_model+".14.prob.4")
            this_neural_model_list.append(neural_model+".14.prob.5")
        '''
        
        for neural_model in neural_models_to_use:
            this_neural_model_list.append(neural_model+".2.prob.1")
            this_neural_model_list.append(neural_model+".2.prob.2")
            this_neural_model_list.append(neural_model+".2.prob.3")
            this_neural_model_list.append(neural_model+".2.prob.4")
            this_neural_model_list.append(neural_model+".2.prob.5")
        active_features =this_handcrafted_features_list+this_neural_model_list
    
    #Normalization
    normalized_df = normalize(df, active_features)
    # Preparation
    Y, X = df['Grade'], normalized_df[[i for i in df.columns if active_features.count(i) == 1]].to_numpy()
    # Find all unique labels
    labels = Y.unique()
    # Map the labels to integer
    mapper = dict()
    for i, v in enumerate(labels):
        mapper[v] = i
    Y = np.array([mapper[i] for i in Y])

    return X, Y, mapper

def CAMEL2SNAKE(name: str) -> str:
    out = list()
    for i in range(len(name) - 1):
        if name[i].isupper() and name[i+1].islower():
            out.append('_' + name[i].lower())
        else:
            out.append(name[i])
    out.append(name[-1])
    return ''.join(out).lstrip('_')