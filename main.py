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


class Mainframe():
    def __init__(self):
        self.sentence_embedder = Embedder()
        self.cols = ["wuc", "wc_code", "updown_ind", "action_taken", "trans_code", "type_maf_code", "type_maint_code", "malfunction_code"]
        wuc_class = Classifier("wuc", "./classifiers/wuc_clf.pkl")
        wc_code_class = Classifier("wc_code", "./classifiers/wc_code_clf.pkl")
        updown_ind_class = Classifier("updown_ind", "./classifiers/updown_ind_clf.pkl")
        action_taken_class = Classifier("action_taken", "./classifiers/action_taken_clf.pkl")
        trans_code_class = Classifier("trans_code", "./classifiers/trans_code_clf.pkl")
        type_maf_code_class = Classifier("type_maf_code", "./classifiers/type_maf_code_clf.pkl")
        type_maint_code_class = Classifier("type_maint_code", "./classifiers/type_maint_code_clf.pkl")
        malfunction_code_class = Classifier("malfunction_code", "./classifiers/malfunction_code_clf.pkl")
        self.classifiers = [wuc_class, wc_code_class, updown_ind_class, action_taken_class, trans_code_class, type_maf_code_class, type_maint_code_class, malfunction_code_class]

    def get_datalog(self, user_input):
        output = {}
        em_input = np.array(self.sentence_embedder.embed(user_input)).tolist()
        for (col, classifier) in zip(self.cols, self.classifiers):
            output[col] = classifier.predict(em_input)
        return output


if  __name__ == "__main__":
    mainframe = Mainframe()
