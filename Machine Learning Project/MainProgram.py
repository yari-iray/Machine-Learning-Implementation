
from math import sqrt
from operator import indexOf
import numpy as np
import matplotlib as mlt
import pandas as pd

DataSplit: list = [0.6, 0.2, 0.2] #A list of the split used in terms of training, validation and test data
K: int = 5 #temporary value, just so we have one right now

def Distance(node1, node2) -> float:
    n: int = len(node1)
    x: float = 0

    for i in range(n):
        x += (node1[i] - node2[i])**2

    return sqrt(x)


def FindNearestNeighbours(node, neighbours: list, k: int):
    # find and return k nearest neighbours to node
    # order the list and take k neares ones

    #obj with 
    distances = [Distance(node, neighbour) for neighbour in neighbours]
    

    # Will rewrite to a node object with neighbours as list, 
    # parameters is a question, maybe a list of features but could be long to implement and non-uniform
    return distances.index(min(distances))


def DistanceFaster(node1, node2) -> float:
    x = sum( [ (node1[i] - node2[i] ** 2) for i in range((len(node1))) ])
    return sqrt(x)

def Load_Dataset():
    CsvData = pd.load('milknew.csv')
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