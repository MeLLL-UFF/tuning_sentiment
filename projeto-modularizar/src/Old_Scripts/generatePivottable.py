from uff.ic.mell.sentimentembedding.datasets.dataset_loader import DataSetLoader
from uff.ic.mell.sentimentembedding.analisador import Analisador

import os
import sys, getopt
import numpy as np
import random
import pandas as pd
import datetime as dt
ts_now = dt.datetime.now()
from time import time
import logging

from uff.ic.mell.sentimentembedding.datasets.dataset import DataSet


def inputsFromComandLine(argv):
    """
        Metodo para pegar dados de input
    """
    try:
        opts, args = getopt.getopt(argv,"hi:d:p:t:",["ifile=", "iDir=", "prefix=", "data="])
    except getopt.GetoptError:
        print ('generatePivottable.py -i <inputfile com dataset> -d <inputDir com resultados da analise> -p <prefix da analise> -t <data do processamento>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('generatePivottable.py -i <inputfile com dataset> -d <inputDir com resultados da analise> -p <prefix da analise> -t <data do processamento>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-d", "--iDir"):
            inputDir = arg
        elif opt in ("-p", "--prefix"):
            prefix = arg
        elif opt in ("-t", "--data"):
            data = arg
    
    return inputfile, inputDir, prefix, data

def main(argv):

    t0 = time()

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='../generatePivottable.log', level=logging.DEBUG)
    SEED = 123

    inputfile, inputDir, prefix, data = inputsFromComandLine(argv)
    print("inputDir: {}, prefix: {}".format(inputfile, inputDir, prefix, data))

    print("iniciando...")
    dsl1 = DataSetLoader(inputfile)
    dsl1.load()
    try:

        datasets = dsl1.datasets

        datasetNames = list(datasets.keys())
        print("imprimindo dataset names...")
        print(datasetNames)

        analisador = Analisador(list(datasets.values()), SEED, "NULO")
        analisador.generate_pivottable(data, inputDir, prefix, inputDir+"/Pivot_tables")

    except Exception as e:
        logger.exception(e)
        raise
    
    print("finalizando...")
    train_time = time() - t0
    print("Duração total: %0.3fs" % train_time)

if __name__ == "__main__":
   main(sys.argv[1:])