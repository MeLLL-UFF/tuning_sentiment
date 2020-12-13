from uff.ic.mell.sentimentembedding.datasets.dataset_loader import DataSetLoader
from uff.ic.mell.sentimentembedding.classificadores.analisador import Analisador

from uff.ic.mell.sentimentembedding.vocabularios.deepmoji_vocabulario import DeepMojiVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.emo2vec_vocabulario import Emo2VecVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.ewe_vocabulario import EWEVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.fasttext_vocabulario import FastTextVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.glove_tw_vocabulario import GloveTWVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.glove_wp_vocabulario import GloveWPVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.sswe_vocabulario import SSWEVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.w2v_araque_vocabulario import W2VAraqueVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.w2v_edin_vocabulario import W2VEdinVocabulario
from uff.ic.mell.sentimentembedding.vocabularios.w2v_gn_vocabulario import W2VGNVocabulario

from uff.ic.mell.sentimentembedding.modelos.modelo_estatico import ModeloEstatico
from uff.ic.mell.sentimentembedding.modelos.modelo_ELMO import ModeloELMO
from uff.ic.mell.sentimentembedding.modelos.modelo_bert import ModeloBert
from uff.ic.mell.sentimentembedding.modelos.modelo_bertweet import Modelobertweet
from uff.ic.mell.sentimentembedding.modelos.modelo_roberta import ModeloRoberta
from uff.ic.mell.sentimentembedding.modelos.modelo_tfidf import ModeloTFIDF
from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer

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
        opts, args = getopt.getopt(argv,"i:m:t:e:o:p:s:",["ifile=", "modelDir=","token=","embeding=" , "oDir=", "ofp=","sSeed="])
    except getopt.GetoptError:
        print ('AvaliaModelos.py -i <inputfile> -g <True/False(default)> -o <outputDir>')
        print ('-m: Language model directory or type model like roberta-case')
        print('-t: Language token directory or type model like roberta-case')
        print('-e: BERT or RoBERTa')
        print ('-p: output file prefix para ser concatenado nos arquivos de saida')
        print('-s: seed')
        sys.exit(2)

    doGridSearch = False

    for opt, arg in opts:
        if opt == '-h':
            print ('AvaliaModelos.py -i <inputfile> -m <LanguageModelDir> -e <embeding RoBERTa or BERT> -o <outputDir> -p <outputFilePrefix> -s <seed>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--oDir"):
            outputDir = arg
        elif opt in ("-p", "--ofp"):
            outputfilePrefix = arg
        elif opt in ("-e", "--ofp"):
            embeding = arg
        elif opt in ("-m", "--lm"):
            modelDir = arg
        elif opt in ("-t", "--lm"):
            token = arg
        elif opt in ("-s", "--sSeed"):
            seed = arg
    return inputfile, outputDir, modelDir, outputfilePrefix, embeding, token,seed

def main(argv):

    t0 = time()

    #SEED = 123
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='sentiment_analysis_tweets.log', level=logging.INFO)

    logger.info("Lendo parametros")
    inputfile, outputDir, modelDir, outputfilePrefix, embedding, token,seed = inputsFromComandLine(argv)
    print("inputFile: {}, outputDir: {}, modelDir: {} outputfilePrefix {} LM {} Token: {}".format(inputfile, outputDir, modelDir, outputfilePrefix,embedding,token))

    print("iniciando...")

    ### set seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    unk_token = "MY_UNKOWN_TOKEN"

    if embedding=="RoBERTa":
        print("runing RoBERTa {}".format(modelDir))
        modelRoberta_tuned = ModeloRoberta(modelDir,token, ModeloTransformer.METHOD.CONTEXT_CONCAT)
        modelRoberta_tuned.originalModel.eval()
        dictionary = {
            "Model": [modelRoberta_tuned],
            'Original': ["roberta-base"]
        }
    elif embedding=="BERT":
        print("runing BERT {}".format(modelDir))
        modelbert_tuned = ModeloBert(modelDir, token, ModeloTransformer.METHOD.CONTEXT_CONCAT)
        modelbert_tuned.originalModel.eval()
        dictionary={
            "Model" : [modelbert_tuned],
            "Original": ["bert-base-uncased"]
        }
    elif embedding=="BERTweet":
        print("runing BERTweet {}".format(modelDir))
        modelBertweetcontext = Modelobertweet(modelDir,token,"CONTEXT")
        modelBertweetcontext.originalModel.eval()
        dictionary = {
            "Model": [modelBertweetcontext],
            'Original': ["BERTweet_base_transformers"]
        }
    elif embedding=="Static":
        vocabDeepMoji = DeepMojiVocabulario("/estudo_orientado/static_embeddings/deepmoji.csv", unk_token, True)
        vocabW2VEdin = W2VEdinVocabulario("/estudo_orientado/static_embeddings/w2v.twitter.edinburgh10M.400d.csv",
                                          unk_token, True)
        vocabW2VAraque = W2VAraqueVocabulario(
            "/estudo_orientado/static_embeddings/twitter_w2vmodel_500d_5mc_ARAQUE_ET_AL", unk_token, True)
        vocabGloveWP = GloveWPVocabulario("/estudo_orientado/static_embeddings/glove.wiki.6B.300d.csv", unk_token, True)
        vocabGloveTW = GloveTWVocabulario("/estudo_orientado/static_embeddings/glove.twitter.27B.200d.csv", unk_token,
                                          True)  # vi palavras com caracteres nao utf-8 aqui
        # print(vocab.getWordEmbedding("rt", "<user>"))
        vocabFastText = FastTextVocabulario("/estudo_orientado/static_embeddings/fastText-wiki-news-300d-1M.txt",
                                            unk_token, True)
        vocabSSWE = SSWEVocabulario("/estudo_orientado/static_embeddings/sswe-u.csv", unk_token, True)
        vocabEmo2Vec = Emo2VecVocabulario("/estudo_orientado/static_embeddings/emo2vec.csv", unk_token,
                                          True)  # vi palavras com caracteres nao utf-8 aqui
        vocabEWE = EWEVocabulario("/estudo_orientado/static_embeddings/ewe.csv", unk_token, True)
        vocabW2VGN = W2VGNVocabulario("/estudo_orientado/static_embeddings/GoogleNews-vectors-negative300.bin",
                                      unk_token, False)

        # print(vocab.getWordEmbedding("the", "</s>"))

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



        modelRobertaStatic = ModeloRoberta("roberta-base","roberta-base", ModeloTransformer.METHOD.STATIC_AVG)
        modelotfidf = ModeloTFIDF()

        modelRobertaStatic.originalModel.eval()

        modelBertStatic = ModeloBert("bert-base-uncased","bert-base-uncased", ModeloTransformer.METHOD.STATIC_AVG)
        modelBertStatic.originalModel.eval()

        modelBertweetSTATIC = Modelobertweet('BERTweet_base_transformers', token, "STATIC")
        modelBertweetSTATIC.BERTweet.eval()
        dictionary = {
            "Model": [modelW2VEdin, modelW2VAraque, modelGloveWP, modelGloveTW, modelFastText,
            modelSSWE, modelEmo2Vec, modelDeepMoji, modelEWE, modelW2VGN, modelRobertaStatic,
            modelotfidf, modelBertStatic, modelBertweetSTATIC]
        }

    logger.info("Carregando dados")
    dsl1 = DataSetLoader(inputfile)
    dsl1.load()
    try:

        datasets = dsl1.datasets

        datasetNames = list(datasets.keys())
        print("imprimindo dataset names...")
        print(datasetNames)

        analisador = Analisador(list(datasets.values()), int(seed), dictionary["Model"],dictionary["Original"])
        #analisar modelos normais
        #analisador.analise(outputDir, outputfilePrefix)

        # analisar modelos indata
        analisador.analise_indata(outputDir, outputfilePrefix)

        # analisar modelos com 21 datasets + indata
        analisador.analise_indata(outputDir, outputfilePrefix, inputfile=inputfile)

    except Exception as e:
        logger.exception(e)
        raise
if __name__ == "__main__":
   main(sys.argv[1:])