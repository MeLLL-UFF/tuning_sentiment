from uff.ic.mell.sentimentembedding.vocabularios.vocabulario import Vocabulario
import pandas as pd
from typing import Dict
import numpy as np

import gensim

class W2VGNVocabulario(Vocabulario):
    def __init__(self, filePath:str, defaultTokenToAdd:str=None, addDefaultToken:bool=False ):
        super().__init__("w2v-GN-vocab", filePath, defaultTokenToAdd, addDefaultToken)
 
    def loadVocabulario(self) -> Dict:
        model = gensim.models.KeyedVectors.load_word2vec_format(self.filePath, binary=True)
        wv = model.wv
        model.add(self.defaultToken, np.zeros(wv.vector_size))
        
        return model