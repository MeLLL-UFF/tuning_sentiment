from uff.ic.mell.sentimentembedding.utils.data_converstion_utils import convert_tensor2array
from uff.ic.mell.sentimentembedding.modelos.modelo import Modelo

import pandas as pd
import numpy as np
import torch
from enum import Enum
from tokenizers import ByteLevelBPETokenizer

class ModeloTransformer(Modelo):

    # média dos tensores da concatenação dos 4 útimos layers - CONTEXT_CONCAT, 
    # média dos tensores do último layer - CONTEXT_LAST 
    # embedding do token [CLS] - CONTEXT_CLS
    # media dos embeddings estaticos das palavras STATIC_AVG
    METHOD = Enum("METHOD", "CONTEXT_CONCAT CONTEXT_LAST CONTEXT_CLS STATIC_AVG")

    def __init__(self, name:str, config, tokenizer, originalModel, embedMethod:METHOD):
        """
           Metodo construtor
           name:                qualquer string que identifique o modelo
           config:              algo do modelo de Transformers. BertConfig() por exemplo
           tokenizer:           tokernizer do modelo. BertTokenizer por exemplo
           originalModel:       modelo propriamente dito. BertModel por exemplo
           embedMethod:         metodo de geracao de embedding das sentencas. Deve ser uma das opcoes do 
                                enum METHOD
        """
        super().__init__(name)
        self.config = config
        self.tokenizer = tokenizer
        self.originalModel = originalModel
        self.embedMethod = embedMethod
    
    def embTexts(self, dataSeries:pd.Series, **kwagars) -> pd.DataFrame:
        '''Função para gerar embedding da BASE DE TWEETS com média dos tensores dos tokens
                Parâmetros:
                    dataSeries: dataframe['tweet']

                return: dataframe com média dos tensores de cada token que perfaz o tweet
        '''
        retorno = []
        
        if (self.embedMethod != ModeloTransformer.METHOD.STATIC_AVG):
            # TODO: Verificar se realmente é necessário definir este tamanho explicitamente
            #       se ficar assim e algum modelo gerar os embeddings de outro tamanho vai dar problema
            if (self.embedMethod == ModeloTransformer.METHOD.CONTEXT_CONCAT):
                #montando array para receber embedding dos tweets do dataframe
                embeddings = np.ndarray((len(dataSeries),3072)) 
            else:
                #montando array para receber embedding dos tweets do dataframe
                embeddings = np.ndarray((len(dataSeries),768)) 
            
            for i, text in enumerate(dataSeries):
                #gerando embeding do text
                tweet = self.get_tweet_embed(text, self.embedMethod) 
                #convertando em um array e inserindo no array criado
                embeddings[i] = convert_tensor2array(tweet.to(device="cpu")) 
            return pd.DataFrame(embeddings)
        else:
            for i, text in enumerate(dataSeries):
                retorno.append(self.transform_sentence_to_avgembword(text))
            return pd.DataFrame(retorno)


    def get_tweet_embed(self, text, method:METHOD, add=True):
        '''Função para gerar embedding do TWEET
            Parâmetros:
                text: tweet a ser tokenizado
                method: conforme enum METHOD
                add: Boolean para adição ou não de tokens especiais, como [CLS]
            return: média dos tensores de cada token que perfaz o tweet
        '''
        self.originalModel.cuda()
        # tokenizar texto, transformar num tensor e enviar para a GPU
        tokens_tensor = torch.tensor([self.tokenizer.encode(text, add_special_tokens=add)]).cuda()

        if (method != ModeloTransformer.METHOD.STATIC_AVG):
            with torch.no_grad():
                out = self.originalModel(tokens_tensor)
                hidden_states = out[2] # selecionando apenas os tensores
                if (method == ModeloTransformer.METHOD.CONTEXT_CONCAT):
                    # get last four layers
                    last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
                    # cast layers to a tuple and concatenate over the last dimension
                    cat_hidden_states = torch.cat(tuple(last_four_layers), dim= -1)
                    # take the mean of the concatenated vector over the token dimension
                    cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1)
                    return cat_sentence_embedding # gerando o embedding da sentença pela média dos embeddings dos tokens concatenados dos 4 últimos layers
                else:
                    if(method == ModeloTransformer.METHOD.CONTEXT_LAST):
                        return torch.mean(hidden_states[-1], dim=1) # gerando o embedding da sentença pela média dos embeddings dos tokens
                    else:
                        if(method == ModeloTransformer.METHOD.CONTEXT_CLS):
                            return hidden_states[-1][:,0,:]

    def transform_sentence_to_avgembword(self, text:str):
        """
            Metodo para gerar embedding das sentencas a partir dos embeddings
            estaticos do modelo usando a media
            Parametros:
                texts: sentenca a ser feito o embedding usando 
                        a media das palavras que a compoe
            Return: 
                retorna um [] com os embeddings das sentencas fazendo a media
                dos embeddings dos tokens que a compoe 
        """

        self.originalModel.cuda()

        # pegando os ids de cada palavra do texto
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        #print(input_ids)
        # pegando o embedding de cada palavra do texto
        #print("#####################")
        ids_tensor = torch.tensor([input_ids]).cuda() #gera um tensor de ids das palavras das sentencas
        #print(ids_tensor.shape)
        embeddings_palavras = self.originalModel.get_input_embeddings()(ids_tensor) # me retorna um tensor de dim 1 x qtdIds x 768
        #print("#####################")
        #print(embeddings_palavras[0])
        # tirando a media e transformando de tensor para array
        #t_stack = torch.stack(embeddings_palavras[0])
        #print("#####################")
        #print(t_stack)
        mean = torch.mean(embeddings_palavras[0], dim=0) # tiro a primeira dimensao do tensor que esta vazia para fazer a media por coluna
        #print("#####################")
        #print (mean)
        #print("#####################")
        mean_arr = convert_tensor2array(torch.unsqueeze(mean, 0)) # recoloco a primeira dimensao para o convert funcionar
        #print (mean_arr)
        return mean_arr

    def tokenize_sentences(self, sentences):
        input_ids = []  # For every sentence...
        for sent in sentences:
            encoded_sent = self.tokenizer.encode(sent,add_special_tokens=True)
            # Add the encoded sentence to the list.
            input_ids.append(encoded_sent)  # Print sentence 0, now as a list of IDs.
        print('Original: ', sentences[0])
        print('Token IDs:', input_ids[0])
        return input_ids

    def train_tokenizer(self,file_path,outDir):
        # Initialize a tokenizer
        tokenizer = ByteLevelBPETokenizer()
        # Customize training
        tokenizer.train(files=file_path, vocab_size=52_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        self.tokenizer=tokenizer
        tokenizer.save(outDir)



