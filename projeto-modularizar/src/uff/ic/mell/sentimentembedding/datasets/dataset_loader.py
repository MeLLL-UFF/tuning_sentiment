
import pandas as pd
from uff.ic.mell.sentimentembedding.utils.datawrangling_utils import DataWrangling
from uff.ic.mell.sentimentembedding.datasets.dataset import DataSet
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from enum import Enum

class DataSetLoader:

    PREPROCESS_TYPE = Enum("PREPROCESS_TYPE", "BASE")

    def __init__(self, filePath:str):
        self.filePath = filePath
        self.datasets = {}
        #self.columns = ["tweet", "dataset", "classe"]
        self.dataWrangling = DataWrangling.getInstance()

    def preprocessAndSave(self, toFile:str, classe_type:str,tweet:int,classe:int,process_type:str):
        """
        metodo para ler o arquivo definidi na inicializacao 
        e salvar em toFile utilizando o preprocessamento definido

        Parametros:
        - toFile: caminho do diretório para salvar a informacao processada
        - process_type: tipo de preprocessamento "base" ou "ekphrasis"
        """
        df = pd.read_csv(self.filePath,encoding='latin-1')
        df = df.dropna()
        if classe_type!='number':
            print("Processing Class...")
            df.iloc[:,int(classe)] = df.iloc[:,int(classe)].apply(self.dataWrangling.transform_class)
        print("Processing: ", process_type)
        if (process_type == "BASE"):
            print("Processing BASE...")
            #df.iloc[:,int(tweet)]= df.iloc[:,int(tweet)].apply(self.preproccess)
            dfinal = pd.DataFrame(columns=['tweet','classe'])
            dfinal['tweet'] = df.iloc[:, int(tweet)].apply(self.preproccess).reset_index(drop=True)
            print(len(dfinal['tweet']))
            print(len(df.iloc[:, int(classe)]))
            dfinal['classe'] = df.iloc[:, int(classe)].reset_index(drop=True)
            print("Saving BASE...")
            print(dfinal.head())
            dfinal.to_csv(toFile, index=False)
            #tfile = open(toFile.split('.')[0] + ".txt", 'a')
            #for i,row in df.iterrows():
            #    tfile.write(row['tweet']+'\n')
            #tfile.close()
        else:
            print("Processing ekphrasis...")
            df.iloc[:,tweet] = df.iloc[:,tweet].apply(DataSetLoader.preproccess_ekphrasis)
            print('salvando arquivo')
            df.to_csv(toFile,index=False)
    
    def load(self):
        """
        Metodo para carregar o arquivo para memoria. Armazena no campo datasets da classe.
        O campo datasets é um dicionário de datasets. Cada chave é o nome do dataset. 
        O valor é um objeto da classe DataSet
        """
        print("\n {} \n".format(self.filePath))
        df = pd.read_csv(str(self.filePath))
        #df = df[self.columns]
        dataset_names = df.dataset.unique()
        print("Data sets Dataloader: ",dataset_names)
        for dataset_name in dataset_names:
            dataset = DataSet(dataset_name, df[df.dataset == dataset_name])
            self.datasets.update({dataset_name : dataset})

    @staticmethod
    def preproccess_ekphrasis(texto:str):
        """
        Método para preprocessar o texto com o método ekphrasis.
        artigo:
        git:

        Parâmetros:
        - texto: texto a ser preprocessado
        """
        text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time', 'url', 'date', 'number'],
            # terms that will be annotated
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                      'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter="twitter",

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector="twitter",

            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )

    def preproccess(self, texto:str):
        """
        Método para preprocessar o texto.

        Parâmetros:
        - texto: texto a ser preprocessado
        """
        temp = self.dataWrangling.changeUserMention(texto, 'someuser')
        temp = self.dataWrangling.changeUrl(temp, 'someurl')
        temp = temp.lower()
        #temp = temp.replace('"', "")
        return temp.strip()

if __name__ == "__main__":
    datasetLoader = DataSetLoader("TESTE")
    print(datasetLoader.preproccess("era CAPS vez http://www.google.com xxx JJJ"))


# END class
