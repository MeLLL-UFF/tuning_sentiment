from uff.ic.mell.sentimentembedding.datasets.dataset_loader import DataSetLoader
from uff.ic.mell.sentimentembedding.modelos.modelo_bert import ModeloBert
from uff.ic.mell.sentimentembedding.modelos.modelo_roberta import ModeloRoberta
from uff.ic.mell.sentimentembedding.modelos.modelo_tfidf import ModeloTFIDF
from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer
from uff.ic.mell.sentimentembedding.classificadores.classificador import ClassificadorWrapper
import os
import sys, getopt
import numpy as np
import random
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import datetime as dt
ts_now = dt.datetime.now()
from time import time

from uff.ic.mell.sentimentembedding.datasets.dataset import DataSet


def inputsFromComandLine(argv):
    """
        Metodo para pegar dados de input
    """
    try:
        opts, args = getopt.getopt(argv,"hi:go:",["ifile=", "gSearch=", "oDir="])
    except getopt.GetoptError:
        print ('BertWOFineTunning.py -i <inputfile> -g <True/False(default)> -o <outputDir>')
        print ('-g: True(para fazer o gridSearch) / False(para nao fazer o gridSearch)')
        sys.exit(2)

    doGridSearch = False

    for opt, arg in opts:
        if opt == '-h':
            print ('BertWOFineTunning.py -i <inputfile> -g <True/False(default)> -o <outputDir>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--oDir"):
            outputDir = arg
        elif opt in ("-g", "--gSearch"):
            doGridSearch = arg
    
    return inputfile, outputDir, doGridSearch

def main(argv):

    t0 = time()

    SEED = 123

    inputfile, outputDir, doGridSearch = inputsFromComandLine(argv)
    print("inputFile: {}, outputDir: {}, doGridSearch: {}".format(inputfile, outputDir, doGridSearch))

    print("iniciando...")
    #dsl = DataSetLoader("projeto1/src/dataset_consolidado.csv")
    #dsl.preprocessAndSave("projeto1/src/dataset_consolidado_pp.csv")

    ### set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    """modelBert1 = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.CONTEXT_CONCAT)
    modelBert1.originalModel.eval()

    modelRoberta1 = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.CONTEXT_CONCAT)
    modelRoberta1.originalModel.eval()

    modelBert2 = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.CONTEXT_LAST)
    modelBert2.originalModel.eval()

    modelRoberta2 = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.CONTEXT_LAST)
    modelRoberta2.originalModel.eval()

    modelBert3 = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.CONTEXT_CLS)
    modelBert3.originalModel.eval()

    modelRoberta3 = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.CONTEXT_CLS)
    modelRoberta3.originalModel.eval()

    modelBert4 = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.STATIC_AVG)
    modelBert4.originalModel.eval()"""

    modelRoberta4 = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.STATIC_AVG)
    modelRoberta4.originalModel.eval()
    

    dictionary={
        "Classifier":[ClassificadorWrapper("Reg_Logistica",
                                        LogisticRegression(solver='liblinear',random_state=SEED,class_weight='balanced',max_iter= 500),
                                        {'penalty' : ['l1', 'l2'], 
                                            'C':(0.2, 0.5, 1.0, 1.5, 2.0, 3.0), 
                                            'solver' : ['liblinear','lbfgs']}),
                   ClassificadorWrapper('Random_Forest',
                                        RandomForestClassifier(random_state=SEED, class_weight="balanced_subsample"),
                                        {'criterion':['gini','entropy'], 'max_depth': [10, 50, 100, None],
                                            'max_features': ['auto', 'sqrt'], 'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10], 'n_estimators': [10, 50, 100]}),
                   ClassificadorWrapper('SVM',
                                        SVC(C=1.0,class_weight='balanced',random_state=SEED),
                                        {'kernel':('linear', 'rbf','poly'), 
                                            'C':(0.2, 0.5, 1.0, 1.5, 2.0, 3.0),
                                            'gamma': (1, 2, 3,'auto','scale'),
                                            'decision_function_shape':('ovo','ovr')}),
                   ClassificadorWrapper("MLPClassifier",
                                        MLPClassifier(random_state=SEED,hidden_layer_sizes=(100, 100)),
                                        {'hidden_layer_sizes': [(50, 50), (50, 100)],
                                            'activation': ['tanh', 'relu'],
                                            'solver': ['sgd', 'adam'],
                                            'alpha': [0.0001, 0.05],
                                            'learning_rate': ['constant','adaptive'],
                                            'early_stopping':(True,False)})],
        "Models" : [
                    #modelBert1, modelRoberta1, 
                    #modelBert2, modelRoberta2, 
                    #modelBert3, modelRoberta3, 
                    #modelBert4, modelRoberta4, 
                    #ModeloTFIDF()]
                    modelRoberta4]
    }

    dsl1 = DataSetLoader(inputfile)
    dsl1.load()
    """data = {'tweet':  ['First value', 'Second value', 'Third Value'],
        'dataset': ['teste', 'teste', 'teste'],
        'classe': [0,0,0]
        }

    df = pd.DataFrame (data, columns = ['tweet','dataset','classe'])
    dataset = DataSet("teste", df)
    dsl1.datasets.update({"teste" : dataset})
    """

    datasets = dsl1.datasets

    datasetNames = list(datasets.keys())
    print("imprimindo dataset names...")
    print(datasetNames)

    for datasetName in datasetNames:
        result=[]
        t1 = time()
        dataset = datasets[datasetName]
        X_train = dataset.getXTrain()
        y_train = dataset.getYTrain()
        print("---------------------- FULL Dataset: {}, {}------------------".format(dataset.name, X_train.shape))
        #print("Proporção entre Classes (positivo - 1 / negativo - 0)")
        #print("DataBase :\n",y_train.value_counts())
        for model in dictionary["Models"]:
            t2 = time()
            print("\n#########################")
            print("\nModel: \n", model.name)

            X_train_emb = model.embTexts(X_train["tweet"])

            for classificadorWrapper in dictionary["Classifier"]:
                t3 = time()
                print("\nCross Validation on {}\n".format(classificadorWrapper.name))
                acc,prec, recall, f1score, F1_macro = classificadorWrapper.kCrossValidation(X_train_emb, y_train, 10, doGridSearch)
                result.append([acc, prec, recall, f1score, F1_macro, classificadorWrapper.name, model.name, datasetName])

                train_time = time() - t3
                print("Duração {}/{}/{}: {:10.0f}s".format(classificadorWrapper.name, model.name, datasetName, train_time))

            train_time = time() - t2
            print("Duração {}/{}: {:10.0f}s".format(model.name, datasetName, train_time))

        final_result_dataset = pd.DataFrame(result, columns=['Accuracy', 'Precision', 'Recall', 'F1score', 'Macro-F1', 'Classifier_Model', "Embedding", "Data_Set"])        
        final_result_dataset.to_csv(outputDir+"/Sem_FT_{}_{}_{}_.csv".format(ts_now.day, ts_now.month, datasetName), index=False)
        train_time = time() - t1
        print("Duração {}: {:10.0f}s".format(datasetName, train_time))

        break
    
    
    print("finalizando...")
    train_time = time() - t0
    print("Duração total: %0.3fs" % train_time)

if __name__ == "__main__":
   main(sys.argv[1:])