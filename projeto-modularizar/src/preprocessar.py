from uff.ic.mell.sentimentembedding.datasets.dataset_loader import DataSetLoader
import os
import sys, getopt
import numpy as np
import random
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
        opts, args = getopt.getopt(argv,"i:o:t:w:c:p:",["ifile=", "ofile=", "clstype=", "tweet=", "classe=","typePP="])
    except getopt.GetoptError:
        print('preprocessar.py -i <inputfile> -o <outputFile> '
              '-p <tipoPreprocess>(Default=base) -w <coluns com tweets> '
              '-c <coluna com classe>'
              '-t <tipo de classe number ou string>')
        sys.exit(2)

    typePP = DataSetLoader.PREPROCESS_TYPE.BASE
    print(opts)
    for opt, arg in opts:
        if opt == '-h':
            print ('preprocessar.py -i <inputfile> -o <outputFile> '
                   '-p <tipoPreprocess>(Default=base) -w <coluns com tweets> '
                   '-c <coluna com classe>'
                   '-t <tipo de classe number ou string>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-t", "--clstype"):
            classe_type = arg
        elif opt in ("-w", "--tweet"):
            tweet = arg
        elif opt in ("-c", "--classe"):
            classe = arg
        elif opt in ("-p", "--typePP"):
            typePP = arg
    
    return inputfile, outputfile, classe_type, tweet, classe,typePP

def main(argv):

    t0 = time()

    inputfile, outputfile, classe_type, tweet, classe,typePP = inputsFromComandLine(argv)
    print("inputFile: {}, outputFile: {}, classe_type: {}, tweet: {}, classe: {},typePP: {}".format(inputfile, outputfile, classe_type, tweet, classe,typePP))

    print("iniciando...")
    dsl = DataSetLoader(inputfile)
    dsl.preprocessAndSave(outputfile, classe_type, tweet, classe, typePP)
    
    print("finalizando...")
    train_time = time() - t0
    print("Duração total: %0.3fs" % train_time)

if __name__ == "__main__":
   main(sys.argv[1:])