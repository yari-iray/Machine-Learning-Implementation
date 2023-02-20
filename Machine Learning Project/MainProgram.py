
from math import sqrt
from operator import indexOf
import numpy as np
import matplotlib as mlt
import pandas as pd
import sklearn

class MainClass:
    self.Dit
    self.Dat

def Distance(node1, node2) -> float:
    n: int = len(node1)
    x: float = 0

    for i in range(n):
        x += (node1[i] - node2[i])**2

    return sqrt(x)


def FindNearestNeighbours(node, neighbours: list, k: int):
    # find and return k nearest neighbours to node
    # order the list and take k neares ones
    distances = []

    for neighbour in neighbours:
        dist = Distance(node, neighbour)
        distances.append(dist)

    return distances.index(min(distances))



def main():
    k = 7
    x = [1,2,3,4,5,6]
    y = [2,3,1,7,8,6]



def DezeFunctieDoetDit():
    camelCase: int = 3

def add(x: int,y: int):
    return x + y

    print("Hello hello")

