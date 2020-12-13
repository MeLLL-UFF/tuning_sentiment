from uff.ic.mell.sentimentembedding.datasets.dataset import DataSet
from uff.ic.mell.sentimentembedding.classificadores.classificador import ClassificadorWrapper
from uff.ic.mell.sentimentembedding.modelos.modelo import Modelo
from uff.ic.mell.sentimentembedding.modelos.modelo_roberta import *
from uff.ic.mell.sentimentembedding.modelos.modelo_bert import *
from uff.ic.mell.sentimentembedding.modelos.modelo_bertweet import *

from time import time
import logging
import pandas as pd
import numpy as np
import datetime as dt
import logging
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import errno
import os
import glob
def check_processed_dataset(path,filesNamePrefix):
    files_processes = glob.glob(path+"/"+filesNamePrefix+"*")
    dataset_processed = [dataset.split("_")[1] for dataset in files_processes]
    return dataset_processed

def encode_dataset(X_train,model):
    tokenizer = model.tokenizer
    embeddings = np.ndarray((len(X_train), 3072))
    with torch.no_grad():
        i=0
        for tweet in list(X_train.values):
            if model.name.split(":")[0]=='Bert' or model.name.split(":")[0]=='Roberta':
                tokens_tensor = torch.tensor(tokenizer(tweet)['input_ids']).unsqueeze(0)
            else:
                tokens_tensor = torch.tensor(tokenizer.tokenize(tweet)).unsqueeze(0)
            out = model.originalModel(tokens_tensor)
            hidden_states = out[2]  # selecionando apenas os tensores
            # get last four layers
            last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
            # cast layers to a tuple and concatenate over the last dimension
            cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
            # take the mean of the concatenated vector over the token dimension
            embeddings[i]= torch.mean(cat_hidden_states, dim=1).detach().numpy()[0]
            i=i+1
        X_embeded = pd.DataFrame(embeddings)
        return X_embeded

class Analisador():
    def __init__(self, datasets:list, seed:int, models:list,model_original:list):
        """
            Inicializador
            Parametros:
                datasets: Array de DataSet
                models: Array de Modelo
        """
        self.logger = logging.getLogger(__name__)
        self.datasets = datasets
        self.seed=seed
        self.classificadores = [ClassificadorWrapper("Reg_Logistica",
                                        LogisticRegression(solver='liblinear',random_state=seed,class_weight='balanced',max_iter= 500),
                                        {'penalty' : ['l1', 'l2'], 
                                            'C':(0.2, 0.5, 1.0, 1.5, 2.0, 3.0), 
                                            'solver' : ['liblinear','lbfgs']}),
                   ClassificadorWrapper('Random_Forest',
                                        RandomForestClassifier(random_state=seed, class_weight="balanced"),
                                        {'criterion':['gini','entropy'], 'max_depth': [10, 50, 100, None],
                                            'max_features': ['auto', 'sqrt'], 'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10], 'n_estimators': [10, 50, 100]}),
                   ClassificadorWrapper('SVM',
                                        SVC(C=1.0,class_weight='balanced',random_state=seed),
                                        {'kernel':('linear', 'rbf','poly'), 
                                            'C':(0.2, 0.5, 1.0, 1.5, 2.0, 3.0),
                                            'gamma': (1, 2, 3,'auto','scale'),
                                            'decision_function_shape':('ovo','ovr')}),
                   ClassificadorWrapper("MLPClassifier",
                                        MLPClassifier(random_state=seed,hidden_layer_sizes=(100, 100),max_iter=300),
                                        {'hidden_layer_sizes': [(50, 50), (50, 100)],
                                            'activation': ['tanh', 'relu'],
                                            'solver': ['sgd', 'adam'],
                                            'alpha': [0.0001, 0.05],
                                            'learning_rate': ['constant','adaptive'],
                                            'early_stopping':(True,False)}),

                    ClassificadorWrapper("XGboost",
                                             xgb.XGBClassifier(objective="binary:logistic", random_state=seed),
                                             {})]

        self.models = models
        self.models_original = model_original
        self.models_tuned=None
        #self.metrics = ['accuracy', 'precision', 'recall', 'f1_macro']
        self.metrics = ['accuracy','f1_macro']


    def analise(self, outputDir:str, filesNamePrefix:str):
        """
            Metodo para fazer a analise gerando arquivos no outputDir
            Parametros:
                outputDir: diretorio onde as analises serao salvas
                filesNamePrefix: prefixo para ser colocado no nome dos arquivos que serão salvos
                                para facilitar assim a identificacao. Os arquivos irão possuir a seguinte
                                regra de formacao {}_{}_{}_{}_.csv (prefixo, dia, mes, nome do dataset)
        """
        ts_now = dt.date
        hoje = ts_now.today()
        self.logger.info("Iniciando experimento")
        processed_data = check_processed_dataset(outputDir,filesNamePrefix)
        print("Datasets já processados: ", processed_data)
        for dataset in self.datasets:
            if dataset.name not in processed_data:
                self.logger.info("Iniciando experimento com data set {}".format(dataset.name))
                result=[]
                t1 = time()
                X_train = dataset.getXTrain()
                y_train = dataset.getYTrain()
                print("---------------------- FULL Dataset: {}, {}------------------".format(dataset.name, X_train.shape))
                self.logger.info("---------------------- FULL Dataset: {}, {}------------------".format(dataset.name, X_train.shape))

                for model in self.models:
                    t2 = time()
                    print("\n######################### \n")
                    print("\nModel: \n", model.name)

                    X_train_emb = model.embTexts(X_train["tweet"])
                    
                    for classificadorWrapper in self.classificadores:
                        t3 = time()

                        self.logger.info("\nCross Validation on {}\n".format(classificadorWrapper.name))

                        print("\nCross Validation on {}\n".format(classificadorWrapper.name))

                        aux_metric = classificadorWrapper.cross_validation(X_train_emb, y_train, 10,
                                                                           metrics=self.metrics)

                        aux_metric.append(classificadorWrapper.name)
                        aux_metric.append(model.name)
                        aux_metric.append(dataset.name)

                        result.append(aux_metric)

                        train_time = time() - t3
                        self.logger.info("Duration {}/{}/{}: {:10.0f}s".format(classificadorWrapper.name, model.name, dataset.name, train_time))

                    train_time = time() - t2

                    self.logger.info("Duration {}/{}: {:10.0f}s".format(model.name, dataset.name, train_time))


                final_result_dataset = pd.DataFrame(result, columns=self.metrics+['Classifier_Model', "Embedding", "Data_Set"])
                final_result_dataset.to_csv(outputDir+"/{}_{}_{}.csv".format(filesNamePrefix, dataset.name, hoje.isoformat()), index=False)
                train_time = time() - t1
                self.logger.info("Finalizado experimento com dataset {}".format(dataset.name))
                self.logger.info("Duration {}: {:10.0f}s".format(dataset.name, train_time))
                print("Duration {}: {:10.0f}s".format(dataset.name, train_time))

        self.generate_pivottable(hoje.isoformat(), outputDir, filesNamePrefix, outputDir+"/Pivot_tables")

    def analise_indata(self, outputDir: str, filesNamePrefix: str, inputfilepath=None):
        """
            Metodo para fazer a analise gerando arquivos no outputDir
            Parametros:
                outputDir: diretorio onde as analises serao salvas
                filesNamePrefix: prefixo para ser colocado no nome dos arquivos que serão salvos
                                para facilitar assim a identificacao. Os arquivos irão possuir a seguinte
                                regra de formacao {}_{}_{}_{}_.csv (prefixo, dia, mes, nome do dataset)
        """
        ts_now = dt.date
        hoje = ts_now.today()
        self.logger.info("Iniciando experimento")
        print(torch.cuda.is_available())
        processed_data = check_processed_dataset(outputDir, filesNamePrefix)
        print("Datasets já processados: ", processed_data)
        for dataset in self.datasets:
            if dataset.name not in processed_data:
                self.logger.info("Iniciando experimento com data set {}".format(dataset.name))
                result = []
                t1 = time()
                X_train = dataset.getXTrain()
                y_train = dataset.getYTrain()
                print(
                    "---------------------- FULL Dataset: {}, {}------------------".format(dataset.name, X_train.shape))
                self.logger.info(
                    "---------------------- FULL Dataset: {}, {}------------------".format(dataset.name, X_train.shape))

                for model in self.models:
                    X = dataset.dataframe['tweet']
                    y = dataset.dataframe['classe']
                    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
                    kf.get_n_splits(X, y)
                    acc_metric=[]
                    f1_metric = []
                    aux_metric=[]
                    for train_index, test_index in kf.split(X, y):
                        X_test = X.iloc[test_index]
                        y_test = y.iloc[test_index]
                        X_train = X.iloc[train_index]
                        if inputfilepath!=None:
                            print("Doing 22DT")
                            filepath = os.path.dirname(inputfilepath)
                            dataset_bench = pd.read_csv(filepath+"/"+str(dataset.name)+".txt",header=0)
                            #print(X_train.reset_index(drop=True))
                            #print(dataset_bench.iloc[:,0])
                            X_train_ = pd.concat([X_train,dataset_bench.iloc[:,0]],ignore_index=True)
                            #print(X_train)
                            #print(type(X_train))
                            print("len dataset:",len(X_train_))
                            X_train_.to_csv('train.txt', index=False)
                        else:
                            X_train.to_csv('train.txt',index=False)
                        y_train = y.iloc[train_index]
                        print("\n######################### \n")
                        print("\nFine tuning Model: \n", model.name)

                        self.models_tuned = model.fine_tune_LM('train.txt', outputDir,self.models_original[0])

                        if model.name.split(":")[0]=='Roberta':
                            model_tuned = ModeloRoberta(outputDir, self.models_original[0], ModeloTransformer.METHOD.CONTEXT_CONCAT)
                        elif model.name.split(":")[0]=='Bert':
                            model_tuned = ModeloBert(outputDir, self.models_original[0],
                                                        ModeloTransformer.METHOD.CONTEXT_CONCAT)
                        else:
                            model_tuned = Modelobertweet(outputDir, self.models_original[0],"CONTEXT")
                        model_tuned.originalModel.eval()

                        print("\n######################### \n")
                        print("\nEncoding Data\n")

                        X_train_emb = model_tuned.embTexts(X_train)
                        X_test_emb = model_tuned.embTexts(X_test)
                        #X_train_emb = encode_dataset(X_train,model_tuned)
                        #X_test_emb = encode_dataset(X_test,model_tuned)
                        acc=[]
                        f1=[]
                        clf_name=[]
                        for classificadorWrapper in self.classificadores:
                            t3 = time()

                            self.logger.info("\nCross Validation on {}\n".format(classificadorWrapper.name))

                            print("\nCross Validation on {}\n".format(classificadorWrapper.name))

                            clf = classificadorWrapper.classificador.fit(X_train_emb, y_train)
                            pred = clf.predict(X_test_emb)

                            acc.append(accuracy_score(y_test,pred))
                            f1.append(f1_score(y_test, pred, average='macro'))
                            clf_name.append(classificadorWrapper.name)


                        acc_metric.append(acc)
                        f1_metric.append(f1)
                    acc_metric = np.array(np.mean(acc_metric,axis=0)).reshape(5,1)
                    f1_metric = np.array(list(np.mean(f1_metric,axis=0))).reshape(5,1)
                    clf_name = np.array(list(clf_name)).reshape(5,1)
                    model_list = np.array([model.name]*len(clf_name)).reshape(5,1)
                    dataset_list = np.array([dataset.name]*len(clf_name)).reshape(5,1)

                    aux_metric = np.array(np.concatenate((acc_metric,f1_metric,clf_name,model_list,dataset_list),axis=1))

                final_result_dataset = pd.DataFrame(aux_metric, columns=self.metrics + ['Classifier_Model', "Embedding",
                                                                                    "Data_Set"])
                final_result_dataset.to_csv(
                    outputDir + "/{}_{}_{}.csv".format(filesNamePrefix, dataset.name, hoje.isoformat()),
                    index=False)

                self.logger.info("Finalizado experimento com dataset {}".format(dataset.name))

        self.generate_pivottable(hoje.isoformat(), outputDir, filesNamePrefix, outputDir + "/Pivot_tables")


    def generate_pivottable(self, dataProcessamento:str, inputPath: str, filesNamePrefix: str, outputPath: str):
        '''Função para montar pivot table com resultados finais
            a partir dos arquivos processados previamente atraves do metodo analise
            Parâmetros:
                dataProcessamento: dia do processamento
                path: caminho para destino do resultado final
        '''
        try:
            os.makedirs(outputPath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        total = pd.DataFrame()

        files_results = glob.glob(inputPath+"/*"+str(filesNamePrefix)+"*.csv")
        for files in files_results:
            final_result_dataset = pd.read_csv(files)
            total = pd.concat([total, final_result_dataset], axis=0)

        total.to_csv(outputPath+"/Final_Result_{}_{}.csv".format(filesNamePrefix, dataProcessamento), index=False)

        classifiers = total["Classifier_Model"].unique()
        for metric in self.metrics:
            for classifier in classifiers:
                df = total[total['Classifier_Model'] == classifier].pivot_table(index='Data_Set', columns=['Embedding'],
                                                                           values=[metric])

                df.to_csv(outputPath+"/pivot_{}_{}_{}_{}.csv".format(classifier,metric, filesNamePrefix, dataProcessamento))

            dfBestRes = total.pivot_table(index='Data_Set', columns=['Embedding'], values=[metric], aggfunc='max')

            dfBestRes.to_csv(outputPath + "/Best_Result_overall_{}_{}_{}.csv".format(metric,filesNamePrefix, dataProcessamento))


        print("Pivot tables salvas!")
