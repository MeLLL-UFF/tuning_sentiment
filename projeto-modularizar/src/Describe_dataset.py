import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import regex
import emoji
import re

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
    df = pd.read_csv("/Users/sergiojunior/sentiment-embeddings/projeto-modularizar/inputs/inputs/sentiment140_preprocessed.csv")
    #del df['Unnamed: 0']
    #df.to_csv("/Users/sergiojunior/sentiment-embeddings/projeto-modularizar/inputs/inputs/sentiment140_preprocessed.csv",index=False)
    #print(df.columns)
