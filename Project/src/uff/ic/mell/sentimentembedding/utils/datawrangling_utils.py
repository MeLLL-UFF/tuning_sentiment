import importlib
twokenize = importlib.import_module("uff.ic.mell.sentimentembedding.utils.ark-twokenize-py.twokenize")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

#nltk.download('stopwords')
#nltk.download('punkt')

import emoji
import re
import os

class DataWrangling():
    __shared_instance = "initialValue"
    
    @staticmethod
    def getInstance():
        if DataWrangling.__shared_instance == "initialValue":
            DataWrangling()
        return DataWrangling.__shared_instance

    def __init__(self):
        if DataWrangling.__shared_instance != "initialValue":
            raise Exception("Só pode haver uma instancia")
        else:
            self.jon_stopwords = []
            __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            with open(os.path.join(__location__, 'stopwords.txt')) as fp: 
                Lines = fp.readlines() 
                for line in Lines: 
                    self.jon_stopwords.append(line.strip())
            #print("self.jon_stopwords: ", self.jon_stopwords)
            DataWrangling.__shared_instance = self
    
    def remove_stopword_jonnathan(self, text:str, tokenizer:str="space"):
        #print("text: ", text)
        #text_tokens = text.split(" ")
        text_tokens = []
        if tokenizer == "twokenize":
            text_tokens = twokenize.tokenizeRawTweetText(text)
        else:
            text_tokens = text.split(" ")
        #print("text_tokens: ", text_tokens)
        tokens_without_sw = [word for word in text_tokens if not word in self.jon_stopwords] 
        #print("tokens_without_sw: ", tokens_without_sw)
        return (" ").join(tokens_without_sw)
    

    def changeUrl(self, text:str, toText:str):
        """
            Substitui urls no texto pelo texto do parâmetro toText
        """
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', toText, text, flags=re.MULTILINE)

    def changeUserMention(self, text:str, toText:str):
        """
            Substitui mencoes a usuários @user no texto pelo texto do parâmetro toText
        """
        #solto no texto depois de algum espaco
        #print(text)
        text = re.sub(r'\s\@\S+', ' '+toText+' ', text, flags=re.MULTILINE)
        #depois de um :
        text = re.sub(r':\@\S+', ' '+toText+' ', text, flags=re.MULTILINE)
        #inicio da linha
        text = re.sub(r'\A@\S+', ' '+toText+' ', text, flags=re.MULTILINE)
        return text

    def transform_class(self, classe:str):
        if (classe=="positive"): return 1
        if (classe=="negative"): return 0
        # não pode nunca chegar neste assert
        assert True, ("Valor da classe não é nem positive e nem negative: {}".format(classe))

def changehashtag(text:str, toText:str):
    """
        Substitui hashtag #text no texto pelo texto do parâmetro toText
    """
    #solto no texto depois de algum espaco
    text = re.sub(r'\s\#[A-Za-z0-9]+\S+', ' '+toText+' ', text, flags=re.MULTILINE)
    #depois de um :
    text = re.sub(r':\#[A-Za-z0-9]+\S+', ' '+toText+' ', text, flags=re.MULTILINE)
    #inicio da linha
    text = re.sub(r'\A#[A-Za-z0-9]+\S+', ' '+toText+' ', text, flags=re.MULTILINE)
    return text

def convert_emojis(text:str):
    """
        Converto emojis pela sua tradução
    """
    return emoji.demojize(text)

def remove_stopword(text:str):
    """
        Retiro stopwords presentes no dicionário padrão da biblio nltk
    """
    tknzr = TweetTokenizer()
    text_tokens = tknzr.tokenize(text)
    #text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()] 
    return (" ").join(tokens_without_sw)

def sentence_length(text):
    emoji_counter = 0
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_counter += 1
            # Remove from the given text the emojis
            text = text.replace(word, '') 

    words_counter = len(text.split())

    return words_counter
    

def emoji_count(text):
    emoji_counter = 0
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_counter += 1
            # Remove from the given text the emojis
            text = text.replace(word, '') 

    return emoji_counter

def mention_count(text):
    #solto no texto depois de algum espaco
    text1 = re.findall(r'\s\@\S+', text)
    #depois de um :
    text2 = re.findall(r':\@\S+', text)
    #inicio da linha
    text3 = re.findall(r'\A@\S+', text)
    return len(text1) + len(text2) + len(text3)

def url_count(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(urls)

def hashtags_count(text):
    #solto no texto depois de algum espaco
    text1 = re.findall(r'\s\#[A-Za-z0-9]+\S+', text)
    #depois de um :
    text2 = re.findall(r':\#[A-Za-z0-9]+\S+', text)
    #inicio da linha
    text3 = re.findall(r'\A#[A-Za-z0-9]+\S+', text)
    return len(text1) + len(text2) + len(text3)

def dataset_describe(df):
    df['Length'] = df['tweet'].apply(sentence_length)
    df['emoji_count'] = df['tweet'].apply(emoji_count)
    df['mention_count'] = df['tweet'].apply(mention_count)
    df['url_count'] = df['tweet'].apply(url_count)
    df['hashtags_count'] = df['tweet'].apply(hashtags_count)
    return df

if __name__ == "__main__":
    print(changeUrl("era uma vez http://www.google.com xxx", "someurl"))
    print(changeUserMention("era uma vez @ricardo xxx", "someuser"))
    print(changehashtag("era uma vez #toaqui xxx", "hashtag"))
    assert transform_class("positive")==1, ("Erro ao transformar classe")
    dw = DataWrangling.getInstance()
    nltk_sw = set(stopwords.words('english'))

    print("len nltk_sw: ", len(nltk_sw))
    print("len jon_sw: ", len(dw.jon_stopwords))

    print("\n#######################")
    print("nltk_sw: ", sorted(nltk_sw))
    print("\n#######################")
    print("jon_sw: ", sorted(dw.jon_stopwords))

    dif_nltk_jon = list(set(nltk_sw) - set(dw.jon_stopwords))
    dif_jon_nltk = list(set(dw.jon_stopwords) - set(nltk_sw))

    print("\n#######################")
    print("len (jon - nltk): ", len(dif_jon_nltk)) 
    print("len (nltk - jon): ", len(dif_nltk_jon))

    print("\n#######################")
    print("(jon - nltk): ", sorted(dif_jon_nltk)) 

    print("\n#######################")
    print("(nltk - jon): ", sorted(dif_nltk_jon))

    print("\n#######################")
    print("ALGUMAS AVALIACOES DO TWOKENIZER...")
    str1 = "I can't see the bird  \u002c  on the table \u0041 !!!!!!! :-D : - D  \u1F600"
    print(str1)
    print(twokenize.tokenizeRawTweetText(str1))

