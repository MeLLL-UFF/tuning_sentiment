from uff.ic.mell.sentimentembedding.classificadores.gridsearch import GridSearch

import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import clone
import numpy as np
import pandas as pd

class ClassificadorWrapper():
    def __init__(self, name, classificador, hyperParams):
        self.name = name
        self.classificador = classificador
        self.hyperParams = hyperParams

    def cross_validation(self, X, y, numCv:int, metrics:list,doGridSearch=False):
        '''Função para realizar o cross validation no data set
            Parâmetros:
                X:dataframe['tweets']
                y:target
                numCv: # de folds
                metrics: metricas a serem coletadas
                doGridSearch: boolean se True faz gridsearch
            return:
                list de metricas na mesma ordem passada no parametro metrics
        '''
        model = self.classificador
        if doGridSearch:
            print("Doing greadsearch {}".format(self.name))
            model = self.getBestModel(X, y)
        metric_vec = []
        scores = cross_validate(model, X, y, cv=numCv,scoring=metrics,return_train_score=False,error_score=0)

        for i, score in enumerate(metrics):
            metric_vec.append(round(np.mean(scores['test_{}'.format(score)]) * 100, 2))
            print("{}: {}".format(score, round(np.mean(scores['test_{}'.format(score)]) * 100, 2)))

        return metric_vec

    def kCrossValidation(self, X, y, numCv:int, doGridSearch:bool):
        '''Função para realizar o cross validation no data set
   
            Parâmetros:
                X:dataframe['tweets']
                y:target
                numCv: # de folds
                doGridSearch: boolean (True para fazer um gridSearch antes do KCrossValidation)
            return: acuracidade média dos folds e precisão média dos folds
        '''
        model = self.classificador
        if doGridSearch:
            model = self.getBestModel(X, y)

        kf = StratifiedKFold(n_splits= numCv, shuffle= True, random_state= 123)
        kf.get_n_splits(X, y) # returns the number of splitting iterations in the cross-validator
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        f1_macro = 0
        for train_index, test_index in kf.split(X, y):
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            
            #clonando modelo de input para garantir que cada fold receberá o modelo original sem influência de outros folds
            #### Fine tuning process ####

            ### Extrair features da base de treino ####

            modelFold = clone(model)
            cf = modelFold.fit(X_train, y_train)

            ### Extrair features da base de teste ###
            tn1, fp1, fn1, tp1 = confusion_matrix(y_test,cf.predict(X_test)).ravel()
            tn = tn + tn1
            fp = fp + fp1
            fn = fn + fn1
            tp = tp + tp1
            modelFold = None
            f1_macro = f1_macro + f1_score(y_test,cf. predict(X_test), average= 'macro') 

        acc = (tp + tn) / (tn + fp + fn + tp)
        f1_macro = f1_macro / 10.0
        if(tp == 0 or (tp + fp) ==0): #evitar "nan" nos resultados pela inexistência de tp e fp
            prec = 0
            recall = 0
            f1score = 0
        else:  
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1score = 2 * ((prec * recall) / (prec + recall))
        print("Mean Accuracy: {0}%".format(round(acc * 100, 2)))
        print("Mean Precision: {0}%".format(round(prec * 100, 2)))
        print("Mean Recall: {0}%".format(round(recall * 100, 2)))
        print("Mean F1-Score: {0}%".format(round(f1score * 100, 2)))
        print("Mean Macro-F1-Score: {0}%".format(round(f1_macro * 100, 2)))
        return round(acc * 100, 2), round(prec * 100, 2), round(recall * 100, 2), round(f1score * 100, 2), round(f1_macro * 100, 2)

    def getBestModel(self, X:pd.DataFrame, y:pd.Series):
        """
            Metodo para retonar o melhor modelo a partir dos hyperparametros definidos no grid
            Parametros:
                X: dataframe com os dados de treinamento
                y: series com os rotulos
        """
        gridSearch = GridSearch(self.classificador, self.hyperParams)
        model = gridSearch.getBestModel(X, y)
        return model