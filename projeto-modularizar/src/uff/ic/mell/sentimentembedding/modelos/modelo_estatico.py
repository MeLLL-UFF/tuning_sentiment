from uff.ic.mell.sentimentembedding.modelos.modelo import Modelo
from uff.ic.mell.sentimentembedding.vocabularios.vocabulario import Vocabulario
import importlib
twokenize = importlib.import_module("uff.ic.mell.sentimentembedding.utils.ark-twokenize-py.twokenize")

import pandas as pd
import torch
from enum import Enum

class ModeloEstatico(Modelo):

    # utiliza a biblioteca https://www.cs.cmu.edu/~ark/TweetNLP/#pos - TWOKENIZER, 
    # separa a sentenca em espacos - SPACE 
    TOKENIZADOR = Enum("TOKENIZADOR", "TWOKENIZER SPACE")

    def __init__(self, name:str, vocabulario:Vocabulario, defaultToken:str, tokenizador:TOKENIZADOR):
        super().__init__(name)
        self.vocabulario = vocabulario
        self.defaultToken = defaultToken
        self.tokenizador = tokenizador
        
    def embTexts(self, dataSeries:pd.Series, **kwagars) -> pd.DataFrame:
        retorno = []
        for i, sentence in enumerate(dataSeries):
            retorno.append(self.avgEmbeddings(self.getWordsEmbeddings(sentence, self.defaultToken)))
        return pd.DataFrame(retorno)

    def avgEmbeddings(self, wordsEmbbeding:[]):
        tensor_words = torch.FloatTensor(wordsEmbbeding)
        mean = torch.mean(tensor_words, dim=0).numpy()
        #print (mean)
        return mean/np.max(np.abs(mean))

    def getWordsEmbeddings(self, sent:str, defaultToken:str) -> list:
        """
            Metodo para retornar os embeddings de varias palavras do vocabulario ou do valor
            default caso a palavra nao faca parte do vocabulario
            Parametros:
                sent: lista de palavras a serem buscadas
                defaultToken: token default para buscar caso a palavra nao seja encontrada
            Exception:
                KeyError: caso nem o defaultToken seja encontrado
            Retorno: retorna uma lista de embeddings
        """
        retorno = []
        #print("({}) sentenca: {}".format(self.name, sent))
        words = []
        if self.tokenizador == ModeloEstatico.TOKENIZADOR.SPACE:
            words = sent.split(" ")
        elif self.tokenizador == ModeloEstatico.TOKENIZADOR.TWOKENIZER:
            words = twokenize.tokenizeRawTweetText(sent)
        totalPalavrasNEncontrada = 0
        for word in words:
            embedding, fromDefault = self.vocabulario.getWordEmbedding(word, defaultToken)
            retorno.append(embedding)
            if (fromDefault):
                totalPalavrasNEncontrada = totalPalavrasNEncontrada + 1
        #print("({}) Total de palavras: {}. Total fora do vocabulario: {}".format(self.name, self.vocabulario.name, len(words), totalPalavrasNEncontrada))

        return retorno
        