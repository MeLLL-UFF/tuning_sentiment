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

from uff.ic.mell.sentimentembedding.modelos.modelo_bert import ModeloBert
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

# https://github.com/google-research/bert/issues/396
# https://github.com/google-research/bert#learning-a-new-wordpiece-vocabulary
# https://github.com/huggingface/transformers/issues/1413

def inputsFromComandLine(argv):
    """
        Metodo para pegar dados de input
    """
    try:
        opts, args = getopt.getopt(argv,"hi:t:o:m:l:",["ifile=", "tokenDir","oDir=","model=","lLog="])
    except getopt.GetoptError:
        print ('Fine_tuning_models.py -i <inputfile> '
               '-t <tokenDir> '
               '-o <outputDir> '
               '-m <model>'
               '-l <loglevel default=2> (1=debug ou 2=info)')
        sys.exit(2)

    log = 2

    for opt, arg in opts:
        if opt == '-h':
            print ('Fine_tuning_models.py -i <inputfile> '
                   '-o <outputDir> '
                   '-l <loglevel default=2> (1=debug ou 2=info)')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--tDir"):
            tokenDir = arg
        elif opt in ("-o", "--oDir"):
            outputDir = arg
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-l", "--lLog"):
            log = int(arg)
    return inputfile, tokenDir, outputDir, model, log


def main(argv):
    SEED = 123
    inputfile, tokenDir,outputDir,model,log = inputsFromComandLine(argv)

    logger = logging.getLogger(__name__)
    logFileName= 'ft_sentiment_token_analysis_tweets_.log'
    loglevel = logging.INFO

    if (log == 1):
        loglevel = logging.DEBUG
    logging.basicConfig(filename=logFileName, level=loglevel)

    logger.info("inputFile: {}, tokenDir{}, outputDir: {} model: {}".format(inputfile, tokenDir ,outputDir,model))
    logger.info("iniciando...")

    ### set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if (model == "RoBERTa"):
        logger.info("running train token roberta")
        print("running train token roberta")
        modelRoberta = ModeloRoberta("roberta-base","roberta-base", ModeloTransformer.METHOD.CONTEXT_CONCAT)
        modelRoberta.train_tokenizer(inputfile,tokenDir)
        print("running ft LM roberta")
        modelRoberta.fine_tune_LM(inputfile, outputDir,tokenizer_path=outputDir)

    if (model == 'BERT'):
        logger.info("running train token bert")
        modelbert = ModeloBert("bert-base-uncased","bert-base-uncased", ModeloTransformer.METHOD.CONTEXT_CONCAT)
        print("running train token bert")
        modelbert.train_tokenizer(inputfile,outputDir)
        print("running ft bert")
        modelbert.fine_tune_LM(inputfile, outputDir,tokenizer_path=outputDir)


if __name__ == "__main__":
   main(sys.argv[1:])