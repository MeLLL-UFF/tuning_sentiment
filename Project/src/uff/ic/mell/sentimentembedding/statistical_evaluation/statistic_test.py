from scipy import stats
import scikit_posthocs as sp
import numpy as np
import pandas as pd
import glob

def friedman_test(dataframe):
    return stats.friedmanchisquare(*[row for index, row in dataframe.T.iterrows()])

def nemenyi_test(dataframe):
    nemenyi = sp.posthoc_nemenyi_friedman(dataframe)
    list_index=[]
    for col in nemenyi.columns:
        list_index.append([col,list(nemenyi[nemenyi[col]<0.05].index),list(nemenyi[nemenyi[col]<0.05][col].values)])
    return  pd.DataFrame(list_index)

def read_dataset(dataframe_path):
    return pd.read_csv(dataframe_path, skiprows=[0,2], sep=",",decimal='.')

PATH='/Users/sergiojunior/sentiment-embeddings-final/Experiment Results/Experiments Results/'
PATH_OUT='/Users/sergiojunior/sentiment-embeddings-final/Experiment Results/Statistical_Reslts/'

#list_experiment=['Static','Transformers','Fine_tuning','Task_Fine_tuning']#'Static','Transformers','Fine_tuning','Task_Fine_tuning'
list_experiment=['Fine_tuning']#'Static','Transformers','Fine_tuning','Task_Fine_tuning'
list_classifiers = ['MLPClassifier','Random_Forest','SVM','XGboost','Reg_Logistica']
list_metrics = ['accuracy','f1_macro']
list_models=['BERT',"RoBERTa",'BERTweet']
for experiment in list_experiment:
    for classifier in list_classifiers:
        for metric in list_metrics:
            print("{}_{}_{}".format(experiment,classifier,metric))
            if experiment=='Static':
                print("Static_embedding")
                df = read_dataset(glob.glob(PATH+experiment+'/Pivot_tables/pivot_'+classifier+'*'+metric+'*.csv')[0])
                print('friedman_test: ',friedman_test(df.iloc[:,1:]))
                nemenyi_test(df.iloc[:,1:]).to_csv(PATH_OUT+"nemenyi_{}_{}_{}.csv".format(experiment,
                                                                                    classifier,
                                                                                    metric))

            if experiment=="Transformers":
                df = read_dataset(glob.glob(PATH+list_models[0]+'/Pivot_tables/pivot_'+classifier+'*'+metric+'*.csv')[0])
                for models in list_models[1:]:
                    print(models)
                    df = df.merge(read_dataset(glob.glob(PATH+models+'/Pivot_tables/pivot_'+classifier+'*'+metric+'*.csv')[0]),
                    how='left',
                    on='Embedding')
                print('friedman_test: ',friedman_test(df.iloc[:,1:]))
                nemenyi_test(df.iloc[:,1:]).to_csv(PATH_OUT+"nemenyi_{}_{}_{}.csv".format(experiment,
                                                                                    classifier,
                                                                                    metric))
            if experiment=='Fine_tuning':
                for models in list_models:
                    print(models)
                    df = pd.read_csv(glob.glob(PATH +'Fine_tuning_Generic_tweets/'+ models + '-1-LM/pivot_' + classifier + '*'+metric+'*.csv')[0])    
                    for k in ['5','05','10','25','50','250','500','1500','6600']:
                        df = df.merge(pd.read_csv(glob.glob(PATH +'Fine_tuning_Generic_tweets/'+ models + '-'+k+'-LM/pivot_' + classifier + '*'+metric+'*.csv')[0]),
                        how='left',
                        on='Embedding',
                        suffixes=("","_"+str(k)))
                    #df_original = pd.read_csv(glob.glob(PATH + models+'/Pivot_tables/pivot_' + classifier + '*'+metric+'*.csv')[0],
                    #                            skiprows=[0,2],sep=",",decimal='.')
                    #df = df.merge(df_original,how='left', on='Embedding')
                    #df.columns=['Embedding','1','5','05','10','25','50','250','500','1500','6600','original']
                    df.columns=['Embedding','1','5','05','10','25','50','250','500','1500','6600']
                    print('friedman_test: ',friedman_test(df.iloc[:,1:]))
                    nemenyi_test(df.iloc[:,1:]).to_csv(PATH_OUT+"nemenyi_{}_{}_{}_{}.csv".format(models,experiment,
                                                                                    classifier,
                                                                                    metric))
            
            if experiment=='Task_Fine_tuning':
                for models in list_models:
                    print(models)
                    df=None
                    df = pd.read_csv(glob.glob(PATH + 'InData/'+models+'-LM/pivot_' + classifier + '*'+metric+'*.csv')[0],sep=",",decimal='.')
                    df.iloc[:,1] = round(df.iloc[:,1]*100,2)
                    for k in ['LOO','22Dt']:
                        df = df.merge(pd.read_csv(glob.glob(PATH + k +'/'+models+'-LM/pivot_' + classifier + '*'+metric+'*.csv')[0],sep=",",decimal='.'),
                        how='left',
                        on='Embedding',
                        suffixes=("","_"+str(k)))
                    df.columns=['Embedding','InData','LOO','22Dt']
                    df['22Dt'] = round(df['22Dt']*100,2)
                    print('friedman_test: ',friedman_test(df.iloc[:,1:]))
                    nemenyi_test(df.iloc[:,1:]).to_csv(PATH_OUT+"nemenyi_{}_{}_{}_{}.csv".format(models,experiment,
                                                                                    classifier,
                                                                                    metric))
            print()