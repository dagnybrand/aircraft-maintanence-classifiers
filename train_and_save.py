import sklearn
import sklearn.model_selection
import sklearn.svm
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tqdm
import os
import pandas as pd

from classifier import Classifier
from embedder import Embedder
from sort_data import get_df

import joblib


def train_and_save():
    datapath = "./../data/"
    dataset = get_df(datapath)
    sentence_embedder = Embedder()
    em_descrep_narrative = np.array(sentence_embedder.embed(dataset['descrep_narrative'])).tolist()
    cols = ["wuc", "wc_code", "updown_ind", "action_taken", "trans_code", "type_maf_code", "type_maint_code", "malfunction_code"]
    wuc_class = Classifier("wuc")
    wc_code_class = Classifier("wc_code")
    updown_ind_class = Classifier("updown_ind")
    action_taken_class = Classifier("action_taken")
    trans_code_class = Classifier("trans_code")
    type_maf_code_class = Classifier("type_maf_code")
    type_maint_code_class = Classifier("type_maint_code")
    malfunction_code_class = Classifier("malfunction_code")
    classifiers = [wuc_class, wc_code_class, updown_ind_class, action_taken_class, trans_code_class, type_maf_code_class, type_maint_code_class, malfunction_code_class]
    count = 0
    for (col, classifier) in zip(cols, classifiers):
        print(type(dataset[col].copy()))
        print(type(np.array(em_descrep_narrative)))
        classifier.split_data(np.array(em_descrep_narrative), dataset[col].copy())
        count += 1
        classifier.train()
        joblib.dump(classifier, f"./classifiers/{col}_clf.pkl")
        classifier.test()
    print('finished')