import pandas as pd

class DataSet:
    def __init__(self, name:str, dataframe: pd.DataFrame):
        self.name = name
        self.dataframe = dataframe
        self.columns = ["tweet", "dataset", "classe"]

    def printDataFrame(self):
        print(self.dataframe)

    def getPositiveClasseLen(self):
        """
        Metodo que retorna a quantidade de tuplas com classe positive
        """
        return len(self.dataframe[self.dataframe[self.columns[2]] == 1])

    def getNegativeClasseLen(self):
        """
        Metodo que retorna a quantidade de tuplas com classe positive
        """
        return len(self.dataframe[self.dataframe[self.columns[2]] == 0])

    def getTotalLen(self):
        """
        Metodo que retorna o total de tuplas do dataset
        """
        return len(self.dataframe)

    def getXTrain(self):
        """
        Metodo para retornar as colunas que não é o rotulo. Retorna uma copia
        """
        return self.dataframe[[self.columns[0], self.columns[1]]].copy()

    def getYTrain(self):
        """
        Metodo para retornar as colunas que é o rotulo. Retorna uma copia
        """
        return self.dataframe[self.columns[2]].copy()


