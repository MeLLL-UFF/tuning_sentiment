import pandas as pd


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


df = pd.read_csv("/sentiment-embeddings/projeto-modularizar/inputs/dataset_consolidado_ppbase_2.csv")
for dataset in list_of_datasets:
    df_filtered = df[df['dataset']!=dataset]
    df_filtered['tweet'].to_csv("/sentiment-embeddings/projeto-modularizar/inputs/{}.txt".format(dataset),header=False,index=False)
    df_filtered_ = df[df['dataset'] == dataset]
    df_filtered_['tweet'].to_csv("/sentiment-embeddings/projeto-modularizar/inputs/{}_evaluation.txt".format(dataset), header=False, index=False)
    df_filtered_ = df[df['dataset'] == dataset]
    df_filtered_.to_csv("/sentiment-embeddings/projeto-modularizar/inputs/{}_evaluation.csv".format(dataset),
                                 header=True, index=False)

