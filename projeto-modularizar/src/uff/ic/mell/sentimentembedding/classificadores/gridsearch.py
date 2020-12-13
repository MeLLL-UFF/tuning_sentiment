import sklearn
from sklearn.model_selection import GridSearchCV
import numpy as np

class GridSearch():
    def __init__(self, classificador, hyperParams):
        self.classificador = classificador
        self.hyperParams = hyperParams

    def getBestModel(self, X, y):
        '''Função para realizar o Grid Search dos classificadores
            Parâmetros:
                X:dataframe['tweets']
                y:target

            return: melhor modelo (Classificador com hyperparametros)
        '''
        # Create grid search using 5-fold cross validation
        clf = GridSearchCV(self.classificador, 
                            self.hyperParams, 
                            cv= 5, verbose= 0, n_jobs= -1, 
                            scoring= ['accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro'], 
                            refit= 'f1_macro') 
        # Fit grid search and return best model
        best_model = clf.fit(X, y)
        print(best_model.best_estimator_.get_params())
  
        for score in best_model.scoring:
            values = best_model.cv_results_['mean_test_%s' % (score)][~np.isnan(best_model.cv_results_['mean_test_%s' % (score)])]
            print("{0}: {1}".format(score,round(np.max(values)*100,3)))
        return best_model
