from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Vocabulario(ABC):

    def __init__(self, name:str, filePath:str, defaultTokenToAdd:str=None, addDefaultToken:bool=False ):
        self.filePath = filePath
        self.name = name
        self.defaultToken = defaultTokenToAdd
        self.lookupTable = self.loadCompleteVocabulario(addDefaultToken)

    def loadCompleteVocabulario(self, addDefaultToken:bool):
        vocab = self.loadVocabulario()
        if (addDefaultToken):
            keys = list(vocab.keys())
            len_values = vocab[keys[0]]
            unknown = np.zeros(len(len_values)) 
            vocab.update({self.defaultToken:unknown})
        return vocab

    @abstractmethod
    def loadVocabulario(self):
        """
            Metodo para ler um arquivo e transformar em um lookup table
            no formato de um dicionario tendo as palavras como chaves e
            embeddings como valores
            Parametros:
                filePath: caminho do arquivo para carregar o volcabulario
        """
        raise NotImplementedError

    def getWordEmbedding(self, word:str, defaultToken:str):
        """
            Metodo para retornar o embedding de uma palavra do vocabulario ou do valor
            default caso a palavra nao faca parte do vocabulario
            Parametros:
                word: palavra a ser buscada
                defaultToken: token default para buscar caso a palavra nao seja encontrada
            Exception:
                KeyError: caso nem o defaultToken seja encontrado
            Retorno: retorna dois valores. Um embedding em forma de lista e um boolean se usou ou nao o default
        """
        retorno = []
        defaultTokenBool = False
        try:
            retorno = self.lookupTable[word]
        except KeyError:
            #se nao encontrar o defaultToken vai dar uma excecao
            #print("({}) Word: '{}' não encontrada. Usando o Token default".format(self.name, word))
            retorno = self.lookupTable[defaultToken]
            defaultTokenBool = True

        return retorno, defaultTokenBool
