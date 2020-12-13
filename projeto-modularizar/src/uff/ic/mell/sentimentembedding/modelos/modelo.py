from abc import ABC, abstractmethod
import pandas as pd
from enum import Enum

class Modelo(ABC):
    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def embTexts(self, texts:pd.Series, **kwagars) ->pd.DataFrame:
        """
            Metodo para transformar textos em embeddings
            Parametros:
                texts: Series de textos para serem transformados em embeddings
                **kwagars: Parametros adicionais que podem ser passados para alguma implementacao especifica
        """
        raise NotImplementedError