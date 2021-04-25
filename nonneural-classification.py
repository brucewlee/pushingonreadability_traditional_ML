import re
import pickle
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from typing import Dict, List, Tuple

# ML Models
import sklearn
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

# ML Utilities
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.utils import shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import max_error

import utils
import handcrafted_features as HF

class ClassificationModel:
    """
        Wrapper Class for Classification Model.

        Parameters
        ----------
            - model: sklearn.Model
                - model to wrap. defaults to LogisticRegression if None
            - class_weight: Dict[int, float]
                - class weights for imbalnced training dataset.
        
        Performs the following:
            - training model (self.train)
            - testing model (self.test)
            - predictions (self.__call__)
            - k-fold validation (self.kfold_validate)
            - train validation (self.train_validate)
            - saving model (self.save)
            - loading model (self.load, class method)
    """
    def __init__(self, model=None, class_weight=None):
        # Define model
        if model is None:
            print("No model provided ... Defaulting to Logistic Regression ...")
            self.model = LogisticRegression(penalty='l1',
                                            solver='saga',
                                            multi_class='multinomial',
                                            max_iter=10000)
        else:
            self.model = model
        assert self.model is not None
        # Provide class weight
        if class_weight is not None:
            self.model.class_weight = class_weight  
        self.model_name = type(self.model).__name__
        self.metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

    def __call__(self, X):
        return self.model.predict(X)

    def train(self, X, Y):
        assert X.shape[0] == Y.shape[0], "There must be the same number of examples. Note that 0-dimension is batch ... "
        self.model.fit(X, Y)
        print("trained")
    
    def test(self, X, Y):
        scoring = ['accuracy', 'precision']
        assert X.shape[0] == Y.shape[0], "There must be the same number of examples. Note that 0-dimension is batch ..."
        pred = self.model.predict(X)
        pred = np.flipud(pred)
        print(pred)
        print(Y)
        acc = sklearn.metrics.accuracy_score(Y, pred)
        f1 = sklearn.metrics.f1_score(Y, pred, average='weighted')
        pre = sklearn.metrics.precision_score(Y, pred, average='weighted')
        rec = sklearn.metrics.recall_score(Y, pred, average='weighted')
        qwk = sklearn.metrics.cohen_kappa_score(Y, pred, weights="quadratic")
        #print(f"Test Accuracy: {acc * 100:.2f}%")
        #print(f"Test F1-score: {f1 * 100:.2f}%")
        #print(f"Test Precision: {pre * 100:.2f}%")
        #print(f"Test Recall: {rec * 100:.2f}%")
        #print(f"Test QWK: {qwk * 100:.2f}%")
        return acc,f1,pre,rec,qwk

    def save(self, path):
        print(f"Saving model to {path} ...")
        pickle.dump(self.model, open(path, 'wb'))
    
    @classmethod
    def load(cls, path):
        print(f"Loading model from {path} ...")
        cls.model = pickle.load(open(path, 'rb'))
    """
    def rf_randomsearch(self, X:np.array, Y:np.array):
        n_estimators = [100, 300, 500, 800, 1200]
        max_depth = [5, 8, 15, 25, 30]
        min_samples_split = [2, 5, 10, 15, 100]
        min_samples_leaf = [1, 2, 5, 10]
        random_grid = dict(n_estimators = n_estimators,
                    max_depth = max_depth,  
                    min_samples_split = min_samples_split, 
                    min_samples_leaf = min_samples_leaf)
        rf_random = GridSearchCV(self.model, random_grid,cv = 3, verbose=1, n_jobs = -1)
        rf_random.fit(X,Y)
        return rf_random.best_params_
    """
    
    def SVC_randomsearch(self, X:np.array, Y:np.array):
        param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
        grid.fit(X,Y)
        return grid.best_params_

    @ignore_warnings(category=ConvergenceWarning)
    def kfold_validate(self, X:np.array, Y:np.array, n_splits:int=10):
        """
            Performs a K-fold validation.

            Parameters
            --------------
                - X: np.array [num_samples, num_features]
                    - numpy array of inputs.
                - Y: np.array [num_samples, ]
                    - numpy array of labels. must not be 1-hot.
            Returns
            --------------
                - None.
        """
        X, Y = shuffle(X, Y)
        scores = cross_validate(self.model, X, Y, scoring=self.metrics, cv=n_splits, n_jobs=4)
        print("metrics over {}-fold validation:".format(n_splits))
        for k, v in scores.items():
            print(k, v.tolist())
            print("average {}:".format(k), sum(v.tolist()) / len(v.tolist()))
            print('*' * 50)

    @ignore_warnings(category=ConvergenceWarning)
    def train_validate(self, X:np.array, Y:np.array):
        """
            Performs a train - validation.

            Parameters
            --------------
                - X: np.array [num_samples, num_features]
                    - numpy array of inputs.
                - Y: np.array [num_samples, ]
                    - numpy array of labels. must not be 1-hot.
            Returns
            --------------
                - None.
        """
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        self.train(X_train, Y_train)
        score = self.test(X_test, Y_test)
        return score

def get_class_error(model, X, Y, mapper):
    """
        Returns the mean and standard deviation error of grade difference 
        between prediction and labels.

        Parameters
        -------------- 
            - model: sklearn.Model
                - classification model to evaluate.
            - X: np.array
                - input features.
            - Y: np.array
                - labels.
            - mapper: Dict[float, int]
                - a dictionary that hashes from grade (float) to multi-class label (int).
        Returns
        --------------
            - None.            
    """
    reverse = {v:k for k,v in mapper.items()}
    Y_grade = np.array([reverse[i] for i in Y])
    Y_pred  = np.array([reverse[i] for i in model(X)])
    print(max_error(Y_grade, Y_pred))
    avg = np.mean(np.abs(Y_grade - Y_pred))
    std = np.std(Y_grade - Y_pred)
    print(f"Error Mean: {avg}, Error St. Dev.: {std}")

def runner(args,k_fold,dataset,neural_model_to_use,this_handcrafted_features_list):
    #train
    df_train = pd.read_csv('Research_Data/'+str(dataset)+'.'+str(k_fold)+'.train.combined.csv')
    
    #test
    df_test = pd.read_csv('Research_Data/'+str(dataset)+'.'+str(k_fold)+'.test.combined.csv')

    X_train, Y_train, mapper = utils.prepare_classification_data(df_train,neural_model_to_use,this_handcrafted_features_list)
    print(len(X_train))
    X_test, Y_test, mapper = utils.prepare_classification_data(df_test,neural_model_to_use,this_handcrafted_features_list)
    print(len(X_test))
    
    MODELS = list()
    if args.logistic:
        MODELS.append(ClassificationModel
                     (LogisticRegression
                     (penalty='l1',
                     dual=False,
                     tol=1e-4,
                     C=1.0,
                     fit_intercept=True,
                     intercept_scaling=1,
                     class_weight=None,
                     random_state=None,
                     solver='saga',
                     max_iter=10000,
                     multi_class='multinomial',
                     verbose=0,
                     warm_start=False,
                     n_jobs=None,
                     l1_ratio=None)))
    if args.perceptron:
        MODELS.append(ClassificationModel(MLPClassifier()))
    if args.svc:
        MODELS.append(ClassificationModel
                     (SVC
                     (C=1, 
                     kernel="poly", 
                     degree=3, 
                     gamma="scale", 
                     coef0=0, 
                     shrinking=True, 
                     probability=False, 
                     tol=1e-3, 
                     cache_size=200, 
                     class_weight=None, 
                     verbose=False, 
                     max_iter=-1, 
                     decision_function_shape="ovr", 
                     break_ties=False, 
                     random_state=None)))
    if args.decision_tree:
        MODELS.append(ClassificationModel(DecisionTreeClassifier()))
    if args.naive_bayes:
        MODELS.append(ClassificationModel(GaussianNB()))
    if args.random_forest:
        MODELS.append(ClassificationModel
                     (RandomForestClassifier
                     (n_estimators=800, 
                     criterion="gini", 
                     max_depth=None, 
                     min_samples_split=2, 
                     min_samples_leaf=1, 
                     min_weight_fraction_leaf=0.0, 
                     max_features="auto", 
                     max_leaf_nodes=None, 
                     min_impurity_decrease=0.0, 
                     min_impurity_split=None, 
                     bootstrap=True, 
                     oob_score=False, 
                     n_jobs=None, 
                     random_state=None, 
                     verbose=0, 
                     warm_start=False, 
                     class_weight=None, 
                     ccp_alpha=0.0, 
                     max_samples=None)))
    if args.gradient_boost:
        MODELS.append(ClassificationModel
                     (xgb.XGBClassifier(
                     eta=0.05,
                     max_depth=9,
                     min_child_weight=0.05,
                     gamma=1,
                     subsample=0.8,
                     objective='softmax')))
    for model in MODELS:
        name = utils.CAMEL2SNAKE(model.model_name)
        print(f"{'=' * 25} RUNNING {name.replace('_', ' ').upper()} ... {'=' * 25}")

        model.train(X_train, Y_train)
        acc,f1,pre,rec,qwk = model.test(X_test, Y_test)
        return acc,f1,pre,rec,qwk
        #model.save(f'checkpoint/classification/{name.lower()}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify models to run ...')
    parser.add_argument(
        '--logistic', '-l',
        action='store_true',
        help="Run logistic regression."
    )
    parser.add_argument(
        '--perceptron', '-p',
        action='store_true',
        help="Run perceptron regression."
    )
    parser.add_argument(
        '--svc', '-s',
        action='store_true',
        help="Run support vector regression."
    )
    parser.add_argument(
        '--decision-tree', '-d',
        action='store_true',
        help="Run decision tree regression."
    )
    parser.add_argument(
        '--naive-bayes', '-n',
        action='store_true',
        help="Run naive bayes."
    )
    parser.add_argument(
        '--random-forest', '-r',
        action='store_true',
        help="Run random forest regression."
    )
    parser.add_argument(
        '--gradient-boost', '-g',
        action='store_true',
        help="Run gradient boosted trees."
    )
    parser.add_argument(
        '--svc-kernel',
        type=str,
        default='rbf',
        help='SVC Kernel to use.'
    )
    args = parser.parse_args()

    # choose dataset
    # 'cambridge' 'weebit' 'onestop'
    dataset = "weebit50"

    # choose neural model to use
    # 'bert' 'roberta' 'bart' 'xlnet' 'none'
    neural_model_to_use = ["bert"]

    # choose handcrafted features to use
    # if FeatureSet_all_HF(), use below
    # handcrafted_features_to_use = HF.FeatureSet_all_as_list_HF()
    # if else, use below
    # handcrafted_features_to_use = [HF.FeatureSet_xxx_HF()]
    handcrafted_features_to_use = [HF.FeatureSet_Total_HF()]
    total_acc_list =[]
    for this_handcrafted_features_list in handcrafted_features_to_use:
        acc_list = []
        f1_list = []
        pre_list = []
        rec_list = []
        qwk_list = []
        if dataset == 'weebit':
            k_fold = [0,1,2,3,4]
            for fold in k_fold:
                acc,f1,pre,rec,qwk = runner(args,fold,'weebit',neural_model_to_use,this_handcrafted_features_list)
                acc_list.append(acc)
                f1_list.append(f1)
                pre_list.append(pre)
                rec_list.append(rec)
                qwk_list.append(qwk)

        if dataset == 'onestop':
            k_fold = [0,1,2]
            for fold in k_fold:
                acc,f1,pre,rec,qwk = runner(args,fold,'onestop',neural_model_to_use,this_handcrafted_features_list)
                acc_list.append(acc)
                f1_list.append(f1)
                pre_list.append(pre)
                rec_list.append(rec)
                qwk_list.append(qwk)

        if dataset == 'cambridge':
            k_fold = [0,1,2,3,4]
            for fold in k_fold:
                acc,f1,pre,rec,qwk = runner(args,fold,'cambridge',neural_model_to_use,this_handcrafted_features_list)
                acc_list.append(acc)
                f1_list.append(f1)
                pre_list.append(pre)
                rec_list.append(rec)
                qwk_list.append(qwk)
        
        if dataset == 'weebit50' or dataset == 'weebit150' or dataset == 'weebit60p' or dataset == 'weebit80p':
            k_fold = [0]
            number_list=[10,20,30,40,50,60,70,80,90,110,120,130,140,150]
            for number in number_list:
                print(number)
                for fold in k_fold:
                    acc,f1,pre,rec,qwk = runner(args,fold,'weebit'+str(number),neural_model_to_use,this_handcrafted_features_list)
                    acc_list.append(acc)
                    f1_list.append(f1)
                    pre_list.append(pre)
                    rec_list.append(rec)
                    qwk_list.append(qwk)
                
                print(acc_list)
                print(f"{'=' * 25} AVERAGING ... {'=' * 25}")
                print("avg acc:" +str(round(float(sum(acc_list) / len(acc_list)), 3)))
                print("avg f1:" +str(round(float(sum(f1_list) / len(f1_list)), 3)))
                print("avg pre:" +str(round(float(sum(pre_list) / len(pre_list)), 3)))
                print("avg rec:" +str(round(float(sum(rec_list) / len(rec_list)), 3)))
                print("avg qwk:" +str(round(float(sum(qwk_list) / len(qwk_list)), 3)))
        #total_acc_list.append((this_handcrafted_features_list[0],round(float(sum(acc_list) / len(acc_list)),3)))
    #total_acc_list.sort(key = lambda x: x[1]) 
    #print(total_acc_list)