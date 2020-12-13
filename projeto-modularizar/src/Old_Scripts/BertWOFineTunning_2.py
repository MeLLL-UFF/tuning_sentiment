from uff.ic.mell.sentimentembedding.datasets.dataset_loader import DataSetLoader
from uff.ic.mell.sentimentembedding.modelos.modelo_bert import ModeloBert
from uff.ic.mell.sentimentembedding.modelos.modelo_roberta import ModeloRoberta
from uff.ic.mell.sentimentembedding.modelos.modelo_tfidf import ModeloTFIDF
from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer
from uff.ic.mell.sentimentembedding.classificadores.classificador import ClassificadorWrapper
from uff.ic.mell.sentimentembedding.classificadores.results_analysis import get_pivottable_result

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
import logging


def inputsFromComandLine(argv):
    """
        Metodo para pegar dados de input
    """
    try:
        opts, args = getopt.getopt(argv, "hi:go:", ["ifile=", "gSearch=", "oDir="])
    except getopt.GetoptError:
        print('BertWOFineTunning.py -i <inputfile> -g <True/False(default)> -o <outputDir>')
        print('-g: True(para fazer o gridSearch) / False(para nao fazer o gridSearch)')
        sys.exit(2)

    doGridSearch = False

    for opt, arg in opts:
        if opt == '-h':
            print('BertWOFineTunning.py -i <inputfile> -g <True/False(default)> -o <outputDir>')
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
    logging.basicConfig(filename='sentiment_analysis_tweets_wGrid.log', level=logging.DEBUG)

    logging.info("Lendo parametros")
    inputfile, outputDir, doGridSearch = inputsFromComandLine(argv)
    print("inputFile: {}, outputDir: {}, doGridSearch: {}".format(inputfile, outputDir, doGridSearch))

    print("iniciando...")

    ### set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logging.info("Criando objetos com modelos de embeddings de tweets")
    modelBert = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.CONTEXT_CONCAT)
    modelBert.originalModel.eval()

    modelRoberta = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.CONTEXT_CONCAT)
    modelRoberta.originalModel.eval()

    modelBertStatic = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.STATIC_AVG)
    modelBertStatic.originalModel.eval()

    modelRobertaStatic = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.STATIC_AVG)
    modelRobertaStatic.originalModel.eval()

    metrics = ['accuracy', 'precision', 'recall', 'f1_macro']

    dictionary = {
        "Classifier": [ClassificadorWrapper("Reg_Logistica",
                                            LogisticRegression(solver='liblinear', random_state=SEED,
                                                               class_weight='balanced', max_iter=500),
                                            {'C': (0.5, 1.0, 1.5)}),
                       ClassificadorWrapper('Random_Forest',
                                            RandomForestClassifier(random_state=SEED, class_weight="balanced"),
                                            {'max_depth': [10, 50, 100, None],
                                             'min_samples_leaf': [1, 2, 4],
                                             'min_samples_split': [2, 5, 10], 'n_estimators': [10, 50, 100]}),
                       ClassificadorWrapper('SVM',
                                            SVC(C=1.0, class_weight='balanced', random_state=SEED),
                                            {'kernel': ('rbf', 'poly'),
                                             'C': (0.5, 1.0, 1.5),
                                             'gamma': ('auto', 'scale')}),
                       ClassificadorWrapper("MLPClassifier",
                                            MLPClassifier(random_state=SEED, hidden_layer_sizes=(100, 100),
                                                          max_iter=500),
                                            {'activation': ['tanh', 'relu'],
                                             'alpha': [0.0001, 0.05],
                                             'learning_rate': ['constant', 'adaptive']})],

        #"Model": [modelBert, modelRoberta, modelBertStatic, modelRobertaStatic, ModeloTFIDF()],
        "Model": [modelBertStatic, modelRobertaStatic, ModeloTFIDF()],
        "Method": 'mean_concat'
    }
    logging.info("Carregando dados")
    dsl1 = DataSetLoader(inputfile)
    dsl1.load()
    try:

        datasets = dsl1.datasets
        del datasets['SemEval16']
        datasetNames = list(datasets.keys())
        print("imprimindo dataset names...")
        print(datasetNames)

        aux_metric = []
        data=['irony','sarcasm']
        logging.info("Iniciando experimento")
        for datasetName in datasetNames:
            logging.info("Iniciando experimento com data seta {}".format(datasetName))
            result = []
            t1 = time()
            dataset = datasets[datasetName]
            X_train = dataset.getXTrain()
            y_train = dataset.getYTrain()
            print("---------------------- FULL Dataset: {}, {}------------------".format(dataset.name, X_train.shape))
            logging.info(
                "---------------------- FULL Dataset: {}, {}------------------".format(dataset.name, X_train.shape))
            # print("Proporção entre Classes (positivo - 1 / negativo - 0)")
            # print("DataBase :\n",y_train.value_counts())
            for model in dictionary["Model"]:
                t2 = time()
                print("\n######################### \n")
                print("\nModel: \n", model.name)
                # logging.info("\nModel: \n", model.name[0])

                X_train_emb = model.embTexts(X_train["tweet"])

                for classificadorWrapper in dictionary["Classifier"]:
                    t3 = time()

                    logging.info("\nCross Validation on {}\n".format(classificadorWrapper.name))
                    print("\nCross Validation on {}\n".format(classificadorWrapper.name))

                    aux_metric = classificadorWrapper.cross_validation(X_train_emb, y_train, 10,
                                                                       metrics=metrics,doGridSearch=doGridSearch)

                    aux_metric.append(classificadorWrapper.name)
                    aux_metric.append(model.name)
                    aux_metric.append(datasetName)

                    result.append(aux_metric)

                    train_time = time() - t3
                    logging.info(
                        "Duration {}/{}/{}: {:10.0f}s".format(classificadorWrapper.name, model.name, datasetName,
                                                              train_time))

                train_time = time() - t2

                logging.info("Duration {}/{}: {:10.0f}s".format(model.name, datasetName, train_time))

            final_result_dataset = pd.DataFrame(result, columns=metrics + ['Classifier_Model', "Embedding", "Data_Set"])
            final_result_dataset.to_csv(
                outputDir + "/Sem_FT_{}_{}_{}_{}_.csv".format('ekphrasis', ts_now.day, ts_now.month, datasetName),
                index=False)
            train_time = time() - t1
            logging.info("Finalizado experimento com data seta {}".format(datasetName))
            logging.info("Duration {}: {:10.0f}s".format(datasetName, train_time))
            print("Duration {}: {:10.0f}s".format(datasetName, train_time))

        get_pivottable_result(datasetNames, ts_now.day, ts_now.month, outputDir, "ekphrasis")
        logging.info("Finalizado pivot table")
        print("finalizando...")
        train_time = time() - t0
        print("Total Duration: %0.3fs" % train_time)
        logging.info("Total Duration: %0.3fs" % train_time)

    except Exception as e:
        logging.exception(e)
        raise


if __name__ == "__main__":
    main(sys.argv[1:])