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
from uff.ic.mell.sentimentembedding.modelos.modelo_bertweet import Modelobertweet
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
import time
import logging



def inputsFromComandLine(argv):
    """
        Metodo para pegar dados de input
    """
    try:
        opts, args = getopt.getopt(argv,"hp:i:o:m:t:l:s:",['pathmodel',"ifile=", "oDir=", "mModel=", "tune=" ,"lLog=",'sSeed='])
    except getopt.GetoptError:
        print ('Fine_tuning_models.py -p <PathOrginModel> -i <inputfile> '
               '-o <outputDir> '
               '-m <model> (1= bert, 2= roberta) '
               '-t <tuning strategy 1=LM, 2=downtask> '
               '-l <loglevel default=2> (1=debug ou 2=info)'
               '-s <seed>')
        sys.exit(2)

    log = 2

    for opt, arg in opts:
        if opt == '-h':
            print ('Fine_tuning_models.py -i <inputfile> '
                   '-o <outputDir> '
                   '-m <modelo> (BERT, RoBERTa) '
                   '-t <tuning strategy LM, Downtask> '
                   '-l <loglevel default=2> (1=debug ou 2=info)')
            sys.exit()
        elif opt in ("-p", "--pathmodel"):
            pathOriginModel = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--oDir"):
            outputDir = arg
        elif opt in ("-t", "--tStrat"):
            tuneStrategy = arg
        elif opt in ("-s","--sSeed"):
            seed=arg
        elif opt in ("-m", "--mModel"):
            model = arg
            if model != 'BERT' and model != 'RoBERTa'and model != 'BERTweet':
                print ("Modelo deve ser Bert ou Roberta ou BERTweet: {}", model)
                sys.exit(2)
        elif opt in ("-l", "--lLog"):
            log = int(arg)
            if model != 1 and model != 2:
                log = 2
    return pathOriginModel,inputfile, outputDir, model, tuneStrategy, log,seed


def main(argv):
    #SEED = 123
    pathOriginModel,inputfile, outputDir, model, tuneStrategy,log,seed = inputsFromComandLine(argv)

    logger = logging.getLogger(__name__)
    logFileName= 'ft_sentiment_analysis_tweets_'+str(model)+'.log'
    loglevel = logging.INFO

    if (log == 1):
        loglevel = logging.DEBUG
    logging.basicConfig(filename=logFileName, level=loglevel)

    logger.info("Model origin: {}, inputFile: {}, outputDir: {}, model: {} Tune strategy {}".format(pathOriginModel,
                                                                                                    inputfile, outputDir,
                                                                                                    model,tuneStrategy))
    logger.info("iniciando...")


    ### set seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    start_time = time.time()
    if (model == "RoBERTa"):
        logger.info("running ft roberta")
        print("running ft roberta")
        modelRoberta = ModeloRoberta(pathOriginModel,pathOriginModel, ModeloTransformer.METHOD.CONTEXT_CONCAT)
        if tuneStrategy == 'LM':
            print("running ft LM roberta")
            modelRoberta.fine_tune_LM(inputfile, outputDir,pathOriginModel)
        else:
            print("running ft Down task roberta")
            ### Tuning Downtask
            epochs = 1
            batch = 64
            learn_rate = 2.5e-5
            weight_decay = 0.001
            modelRoberta.fine_tune_downtask(inputfile, epochs, batch, outputDir, learn_rate,weight_decay)

    elif  (model == 'BERT'):
        logger.info("running ft bert")
        print("running ft bert")
        modelbert = ModeloBert(pathOriginModel,pathOriginModel, ModeloTransformer.METHOD.CONTEXT_CONCAT)
        modelbert.fine_tune_LM(inputfile, outputDir,pathOriginModel)
    else:
        logger.info("running ft bertweet")
        print("running ft bertweet")
        modelbertweet = Modelobertweet(pathOriginModel,pathOriginModel, 'CONTEXT',True)
        modelbertweet.fine_tune_LM(inputfile, outputDir,pathOriginModel)

    Total_time=time.time() - start_time
    f = open(outputDir+"/tempo_{}.txt".format(outputDir.split('/')[-1]), "a")
    f.write("")
    f.write(str(Total_time))
    f.close()
if __name__ == "__main__":
   main(sys.argv[1:])