from dataclasses import dataclass
import math
import pandas as pd
import numpy as np
from kNN import KNN
from Network import NeuralNetwork

DataSplit: dict = {"training": 0.6, "validation": 0.2, "test": 0.2 }
K: int = 5
Dataset: str = "milknew.csv"
np.random.seed(1)

class DataFunctions:
    def LoadDataset():
        CsvData = pd.read_csv(Dataset)
        Length: int = len(CsvData) #needed to split the data into parts

        #Split data into train, validation, test
        TrainData = CsvData[0: int(Length * DataSplit["training"])] 
        ValidationData = CsvData[int(Length * DataSplit["training"]): int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"])]
        TestData = CsvData[int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"]): Length] 

        return TrainData, ValidationData, TestData


    def NormalizeData(DataSet):
        n: int = len(DataSet.columns) #Number of columns

        #the last column contains the grade, which doesn't need to be normalized as it is a classification and not a number
        #loop over all columns and normalize them
        for i in range(n - 1): 
            LowerBound: int = min(DataSet.iloc[:,i]) #Column min
            UpperBound: int = max(DataSet.iloc[:,i]) #Column max

            DataSet.iloc[:,i] = (DataSet.iloc[:,i] - LowerBound) / (UpperBound - LowerBound) #Using this function, the column will be normalized to be between 0 and 1
    
        return DataSet

def Main():
    TrainData, ValidationData, TestData = DataFunctions.LoadDataset()
    TrainData = DataFunctions.NormalizeData(TrainData)
    ValidationData = DataFunctions.NormalizeData(ValidationData)
    TestData = DataFunctions.NormalizeData(TestData)


    #create a neural network instance with 2 hidden layers of 1 neuron each
    Network = NeuralNetwork(TrainData, [7,3,4,1,8,4])


    ##### for testing purposes only, try to see if the training data nets the same problems as the
    ##### testdata
    resultValuesForTrainingData = []
    for i in range(len(Network.Input)):
        values = Network.ComputeNeuralNetwork(i)

        resultValuesForTrainingData.append(values.AbsoluteOutput)

    expectedResults = Network.ExpectedOutput

    print('old weights')
    print(Network.Weights)
    #print('new weights')
    #print(Network.BackPropagate(0))

    #Network.TestNetwork(TestData)

    

if __name__ == "__main__":
    Main()

