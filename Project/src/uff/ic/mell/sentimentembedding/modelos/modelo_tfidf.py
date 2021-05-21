from uff.ic.mell.sentimentembedding.modelos.modelo import Modelo
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class ModeloTFIDF(Modelo):
    def __init__(self):
        super().__init__("TFIDF")

    def embTexts(self, dataSeries:pd.Series, **kwagars) -> pd.DataFrame:
        '''Montar dataframe com vetores dos tweets pelo Bow com TFDF
            Parametros:
                corpus: pd.Series com os tweets
            return: dataframe com os vetores dos textos
        '''
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataSeries)
        return pd.DataFrame(X.todense())