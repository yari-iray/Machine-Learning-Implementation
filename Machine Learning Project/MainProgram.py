
from math import sqrt
from operator import indexOf
import numpy as np
import matplotlib as mlt
import pandas as pd

DataSplit: list = [0.6, 0.2, 0.2]

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

    Length = len(CsvData)
    TrainData = CsvData[0: int(Length * DataSplit[0])]
    ValidationData = CsvData[int(Length * DataSplit[0]): int(Length * DataSplit[0]) + int(Length * DataSplit[1])]
    TestData = CsvData[int(Length * DataSplit[0]) + int(Length * DataSplit[1]): Length]
    pass


def Main():
    pass