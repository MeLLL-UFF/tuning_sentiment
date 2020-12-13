from uff.ic.mell.sentimentembedding.vocabularios.vocabulario import Vocabulario
import pandas as pd
from typing import Dict

class FastTextVocabulario(Vocabulario):
    def __init__(self, filePath:str, defaultTokenToAdd:str=None, addDefaultToken:bool=False ):
        super().__init__("fastText-vocab", filePath, defaultTokenToAdd, addDefaultToken)

    def loadVocabulario(self) -> Dict:
        #arquivo com colunas separados por espaco, sem header, 
        # com indice na primeira coluna e nao troca qualquer key por NaN (tem as palavaras nan e null no vocabulario)
        df = pd.read_csv(self.filePath, sep=" ", header=None, 
                            index_col=0, keep_default_na=False,
                            quoting=3)
        print(df.head(5))
        lookupTable = {}
        for ix, row in df.iterrows():
            #print(index, row)
            values = row.values
            lookupTable.update({ix : values})
        return lookupTable