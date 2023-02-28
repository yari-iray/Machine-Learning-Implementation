
from math import sqrt
from operator import indexOf
import numpy as np
import matplotlib as mlt
import pandas as pd

DataSplit: list = [0.6, 0.2, 0.2] #A list of the split used in terms of training, validation and test data
K: int = 5 #temporary value, just so we have one right now
DataSet: str = 'milknew.csv' #The dataset currently used

def Distance(node1, node2) -> float:
    n: int = len(node1)
    x: float = 0

    for i in range(n):
        x += (node1[i] - node2[i])**2

    return sqrt(x)


def FindNearestNeighbours(DataFrame, Node, k: int):
    # find and return k nearest neighbours to node
    # order the list and take k neares ones
    NeighbourList = DataFrame #Create a separate Dataframe to store the distance to the given Node
    NeighbourList['Distance'] = 99999 #Set the distance to a very high number at first.

    #Find all the distances to the points in the dataframe from Node.
    for i in range(1, DataFrame.shape[0]-1):
        NeighbourList.iloc[i,NeighbourList.shape[1]-1] = Distance(Node, DataFrame.iloc[i,:])
    
    NeighbourList = NeighbourList.sort_values(by='distance', ascending=True) #Sort the dataframe by distance
    NeighbourList = NeighbourList.iloc[0:k,:] #Keep only the k lowest distances, and return them.
   
    return NeighbourList


def DistanceFaster(node1, node2) -> float:
    x = sum( [ (node1[i] - node2[i] ** 2) for i in range((len(node1))) ])
    return sqrt(x)

class Data:
    def LoadDataset():
        CsvData = pd.load(DataSet)
        Length: int = len(CsvData) #Needed to make a good split of data

        #The data has to be split into training, validation and test data using predefined percentages.
        TrainData = CsvData[0: int(Length * DataSplit[0])] 
        ValidationData = CsvData[int(Length * DataSplit[0]): int(Length * DataSplit[0]) + int(Length * DataSplit[1])]
        TestData = CsvData[int(Length * DataSplit[0]) + int(Length * DataSplit[1]): Length] 

        return TrainData, ValidationData, TestData


    def NormalizeData(DataSet):
        n: int = len(DataSet.columns) #Find the columns so we know how far to iterate

        for i in range(n-1): #range is n-1 because the last column does not contain numbers, but the class.
            LowerBound: int = min(DataSet.iloc[:,i]) #Find the minimum value in the current column
            UpperBound: int = max(DataSet.iloc[:,i]) #Find the maximum value in the current column

            DataSet.iloc[:,i] = (DataSet.iloc[:,i] - LowerBound) / (UpperBound - LowerBound) #Using this function, the column will be normalized to be between 0 and 1
    
        return DataSet

def Main():
    pass