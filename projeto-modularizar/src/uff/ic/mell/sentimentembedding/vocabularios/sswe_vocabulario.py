from uff.ic.mell.sentimentembedding.vocabularios.vocabulario import Vocabulario
import pandas as pd
from typing import Dict

class SSWEVocabulario(Vocabulario):
    def __init__(self, filePath:str, defaultTokenToAdd:str=None, addDefaultToken:bool=False ):
        super().__init__("SSWE-vocab", filePath, defaultTokenToAdd, addDefaultToken)

    def loadVocabulario(self) -> Dict:
        #df = pd.read_csv(self.filePath, sep=" ", header=None, index_col=0, keep_default_na=False)
        df = pd.read_csv(self.filePath, sep="\t", header=None, index_col=0, keep_default_na=False)
        print(df.head(5))
        lookupTable = {}
        for ix, row in df.iterrows():
            #print(index, row)
            values = row.values
            lookupTable.update({ix : values})
        
        return lookupTable