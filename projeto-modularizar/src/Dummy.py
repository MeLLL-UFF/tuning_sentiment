from transformers import RobertaTokenizer,BertTokenizer,AutoTokenizer
from transformers import RobertaTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import glob
import numpy as np
from transformers import GPT2Tokenizer

'''print("oops!! pelosi & dems admit numbers submitted to cbo are false! someurl #tcot #tlot #sgp #hcr #p2")
print()
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
normal = tokenizer.encode("oops!! pelosi & dems admit numbers submitted to cbo are false! someurl #tcot #tlot #sgp #hcr #p2")
print(tokenizer.convert_ids_to_tokens(normal))
print()
tokenizer2 = BertweetTokenizer.from_pretrained("vinai/bertweet-base")
trained = tokenizer2.encode("oops!! pelosi & dems admit numbers submitted to cbo are false! someurl #tcot #tlot #sgp #hcr #p2")
print(trained)
print(tokenizer2.convert_ids_to_tokens(trained))'''

list_of_datasets=['HCR',
                  'Narr-KDML-2012',
                  'STS-gold',
                  'SemEval13',
                  'SemEval15-Task11',
                  'SemEval16',
                  'SemEval17-test',
                  'SemEval18',
                  'SentiStrength',
                  'Target-dependent',
                  'VADER',
                  'archeage',
                  'debate08',
                  'hobbit',
                  'iphone',
                  'irony',
                  'movie',
                  'ntua',
                  'person',
                  'sanders',
                  'sarcasm',
                  'sentiment140']
'''for strat in ['LM','InData','Union']:
    for model in ["BERT", "RoBERTa", 'BERTweet']:
        df = None
        for dataset in list_of_datasets:
            if strat == 'Union':
                file = glob.glob('./{}{}*/tempo_*.txt'.format(model, strat))
                print(file)
                df = open(file[0], 'r')
                tempo.append([model, dataset, df.readline().replace('\n', ''), strat])
                break
            else:
                file = glob.glob('./{}{}{}*/tempo_*.txt'.format(model, dataset,strat))
                print(file)
                df = open(file[0],'r')
                tempo.append([model,dataset,df.readline().replace('\n',''),strat])

pd.DataFrame(tempo).to_csv('tempo_Indata.csv',index=False)

df=pd.read_csv('/Users/sergiojunior/sentiment-embeddings/scripts_artigo/tempo_Indata.csv')
print(df.head())
df.columns=['Embedding','Dataset','Time','Strategy']
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
#print(df['Time'].mean())
dff = df[['Strategy','Embedding','Time']].groupby(by=['Strategy','Embedding']).mean()
dff.to_csv('time.csv')'''


