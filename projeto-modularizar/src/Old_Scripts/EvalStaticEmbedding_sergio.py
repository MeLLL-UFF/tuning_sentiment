from uff.ic.mell.sentimentembedding.datasets.dataset_loader import DataSetLoader
from uff.ic.mell.sentimentembedding.analisador import Analisador
from uff.ic.mell.sentimentembedding.modelos.modelo_bert import ModeloBert
from uff.ic.mell.sentimentembedding.modelos.modelo_roberta import ModeloRoberta
from uff.ic.mell.sentimentembedding.modelos.modelo_tfidf import ModeloTFIDF
from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer
#from uff.ic.mell.sentimentembedding.modelos.modelo_ELMO import ModeloELMO

import sys, getopt
import numpy as np
import random
import torch
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
        opts, args = getopt.getopt(argv,"hi:go:p:",["ifile=", "gSearch=", "oDir=", "ofp="])
    except getopt.GetoptError:
        print ('BertWOFineTunning.py -i <inputfile> -g <True/False(default)> -o <outputDir>')
        print ('-g: True(para fazer o gridSearch) / False(para nao fazer o gridSearch)')
        print ('-p: output file prefix para ser concatenado nos arquivos de saida')
        sys.exit(2)

    doGridSearch = False

    for opt, arg in opts:
        if opt == '-h':
            print ('BertWOFineTunning.py -i <inputfile> -g <True/False(default)> -o <outputDir> -p <outputFilePrefix>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--oDir"):
            outputDir = arg
        elif opt in ("-p", "--ofp"):
            outputfilePrefix = arg
        elif opt in ("-g", "--gSearch"):
            doGridSearch = arg

    return inputfile, outputDir, doGridSearch, outputfilePrefix

def main(argv):

    t0 = time()

    SEED = 123
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='../sentiment_analysis_tweets.log', level=logging.DEBUG)

    logger.info("Lendo parametros")
    print(inputsFromComandLine(argv))
    inputfile, outputDir, doGridSearch, outputfilePrefix = inputsFromComandLine(argv)
    print("inputFile: {}, outputDir: {}, doGridSearch: {} outputfilePrefix: {}".format(inputfile, outputDir,
                                                                                       doGridSearch, outputfilePrefix))

    print("iniciando...")

    ### set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    modelBert = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.CONTEXT_CONCAT)
    modelBert.originalModel.eval()

    modelRoberta = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.CONTEXT_CONCAT)
    modelRoberta.originalModel.eval()

    modelBertStatic = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.STATIC_AVG)
    modelBertStatic.originalModel.eval()

    modelRobertaStatic = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.STATIC_AVG)
    modelRobertaStatic.originalModel.eval()

    #modelELMO = ModeloELMO(OutputType="default")


    #dictionary={
    #    "Model": [modelBertStatic, modelRobertaStatic, ModeloTFIDF()],
    #    "Method": 'mean_concat'
    #}

    dictionary = {
            "Model": [modelBert,modelRoberta]
    }

    logger.info("Carregando dados")
    dsl1 = DataSetLoader(inputfile)
    dsl1.load()
    try:

        datasets = dsl1.datasets

        datasetNames = list(datasets.keys())
        print("imprimindo dataset names...")
        print(datasetNames)

        analisador = Analisador(list(datasets.values()), SEED, dictionary["Model"])
        analisador.analise(outputDir, outputfilePrefix)

    except Exception as e:
        logger.exception(e)
        raise
if __name__ == "__main__":
   main(sys.argv[1:])