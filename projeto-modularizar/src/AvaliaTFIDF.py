from uff.ic.mell.sentimentembedding.datasets.dataset_loader import DataSetLoader
from uff.ic.mell.sentimentembedding.classificadores.analisador import Analisador


from uff.ic.mell.sentimentembedding.modelos.modelo_estatico import ModeloEstatico

from uff.ic.mell.sentimentembedding.modelos.modelo_tfidf import ModeloTFIDF

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
        print ('AvaliaModelos.py -i <inputfile> -g <True/False(default)> -o <outputDir>')
        print ('-g: True(para fazer o gridSearch) / False(para nao fazer o gridSearch)')
        print ('-p: output file prefix para ser concatenado nos arquivos de saida')
        sys.exit(2)

    doGridSearch = False

    for opt, arg in opts:
        if opt == '-h':
            print ('AvaliaModelos.py -i <inputfile> -g <True/False(default)> -o <outputDir> -p <outputFilePrefix>')
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
    logging.basicConfig(filename='sentiment_analysis_tweets.log', level=logging.DEBUG)

    logger.info("Lendo parametros")
    inputfile, outputDir, doGridSearch, outputfilePrefix = inputsFromComandLine(argv)
    print("inputFile: {}, outputDir: {}, doGridSearch: {} outputfilePrefix {}".format(inputfile, outputDir, doGridSearch, outputfilePrefix))

    print("iniciando...")

    ### set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    unk_token = "MY_UNKOWN_TOKEN"

    '''vocabDeepMoji = DeepMojiVocabulario("/estudo_orientado/static_embeddings/deepmoji.csv", unk_token, True)
    vocabW2VEdin = W2VEdinVocabulario("/estudo_orientado/static_embeddings/w2v.twitter.edinburgh10M.400d.csv", unk_token, True)
    vocabW2VAraque = W2VAraqueVocabulario("/estudo_orientado/static_embeddings/twitter_w2vmodel_500d_5mc_ARAQUE_ET_AL", unk_token, True)
    vocabGloveWP = GloveWPVocabulario("/estudo_orientado/static_embeddings/glove.wiki.6B.300d.csv", unk_token, True)
    vocabGloveTW = GloveTWVocabulario("/estudo_orientado/static_embeddings/glove.twitter.27B.200d.csv", unk_token, True) #vi palavras com caracteres nao utf-8 aqui
    #print(vocab.getWordEmbedding("rt", "<user>"))
    vocabFastText = FastTextVocabulario("/estudo_orientado/static_embeddings/fastText-wiki-news-300d-1M.txt", unk_token, True)
    vocabSSWE = SSWEVocabulario("/estudo_orientado/static_embeddings/sswe-u.csv", unk_token, True)
    vocabEmo2Vec = Emo2VecVocabulario("/estudo_orientado/static_embeddings/emo2vec.csv", unk_token, True) #vi palavras com caracteres nao utf-8 aqui
    vocabEWE = EWEVocabulario("/estudo_orientado/static_embeddings/ewe.csv", unk_token, True)
    vocabW2VGN = W2VGNVocabulario("/estudo_orientado/static_embeddings/GoogleNews-vectors-negative300.bin", unk_token, False)

    #print(vocab.getWordEmbedding("the", "</s>"))

    modelW2VEdin = ModeloEstatico("W2VEdin", vocabW2VEdin, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelW2VAraque = ModeloEstatico("W2VAraque", vocabW2VAraque, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelGloveWP = ModeloEstatico("GloveWP", vocabGloveWP, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelGloveTW = ModeloEstatico("GloveTW", vocabGloveTW, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelFastText = ModeloEstatico("FastText", vocabFastText, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelSSWE = ModeloEstatico("SSWE", vocabSSWE, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelEmo2Vec = ModeloEstatico("Emo2Vec", vocabEmo2Vec, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelDeepMoji = ModeloEstatico("DeepMoji", vocabDeepMoji, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelEWE = ModeloEstatico("EWE", vocabEWE, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)
    modelW2VGN = ModeloEstatico("W2V-GN", vocabW2VGN, unk_token, ModeloEstatico.TOKENIZADOR.TWOKENIZER)

    modelBertStatic = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.STATIC_AVG)
    modelBertStatic.originalModel.eval()
    modelRobertaStatic = ModeloRoberta("roberta-base", ModeloTransformer.METHOD.STATIC_AVG)
    modelRobertaStatic.originalModel.eval()'''

    #modelBert = ModeloBert("bert-base-uncased", ModeloTransformer.METHOD.CONTEXT_CONCAT)
    #modelBert.originalModel.eval()

    #modelRoberta_tuned = ModeloRoberta('./test/saved_model/', ModeloTransformer.METHOD.CONTEXT_CONCAT)
    #modelRoberta_tuned.originalModel.eval()

    modelotfidf = ModeloTFIDF()

    dictionary={
        "Model" : [modelotfidf]
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