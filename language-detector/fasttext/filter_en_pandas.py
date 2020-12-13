import pandas as pd
import csv
import datetime as dt
from time import time
import os
import sys, getopt
import numpy as np

def main(argv):

    t1 = time()

    inputfile, outputfile = inputsFromComandLine(argv)
    print("inputFile: {}, outputfile: {}".format(inputfile, outputfile))


    print ("iniciando....")

    #df_source = pd.read_csv('teste.csv', sep="\t", header=None, dtype=str, keep_default_na=False)
    df_source = pd.read_csv(inputfile, sep="\t", header=None, quoting=csv.QUOTE_NONE)
    df_source.columns = ["label",  "text"]
    #df_source['linenum'] = np.arange(len(df_source))

    #print(df_source.head(10))

    #df_1 = df_source.loc[df_source["linenum"].isin([8564, 8565, 8566, 8567, 8568, 8569, 30992, 34217, 64229, 74732, 78405, 78599, 94330, 99298, 1101134])]["text"]
    #df_1.to_csv("result1")

    print("filtrando e gerando novo arquivo")
    df_en = df_source.loc[df_source["label"].astype(str).str.contains("__label__en")]

    print(df_en.head(10))

    #length before adding row 
    length1 = len(df_en) 
    print (length1)

    #df_2 = df_en.loc[df_en["linenum"].isin([8567, 30992, 34217, 64229, 74732, 78405, 78599, 94330, 99298, 1101134])]["text"]
    #df_2.to_csv(outputfile)

    # dropping duplicate values 
    df_en_un = df_en.drop_duplicates(subset ="text")

    length1 = len(df_en_un) 
    print (length1)

    x = df_en_un["text"].astype(str).str.contains("__label__")
    print(x.unique())

    df_teste = df_en_un.loc[x]
    #print(df_teste.head(10))
    length1 = len(df_teste) 
    print (length1)

    df_en_un_text = df_en_un["text"]
    df_en_un_text.to_csv(outputfile, index=False, header=None)

    train_time = time() - t1
    print("Duration: {:10.0f}s".format(train_time))


def inputsFromComandLine(argv):
    """
        Metodo para pegar dados de input
    """
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=", "ofile="])
    except getopt.GetoptError:
        print ('preprocessar.py -i <inputfile> -o <outputFile>')
        sys.exit(2)

    for opt, arg in opts:
        print("opt: {}, arg: {}".format(opt, arg))
        if opt == '-h':
            print ('filter_en_pandas.py -i <inputfile> -o <outputFile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    
    return inputfile, outputfile

if __name__ == "__main__":
   main(sys.argv[1:])