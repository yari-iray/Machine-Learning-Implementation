import math
import pandas as pd

class KNN:
    def __init__(self, k: int):
        self.K = k

    # find and return k nearest neighbours to node by ordering list and taking the k nearest ones
    def FindNearestNeighbours(self, Data: pd.DataFrame, Node):

        NeighbourList = Data #Create a separate Dataframe to store the distance to the given Node
        NeighbourList['Distance'] = math.inf #Initialize as infinity

        #Find all the distances to the points in the dataframe from Node.
        for i in range(0, Data.shape[0]):
            NeighbourList.iloc[i,NeighbourList.shape[1]-1] = self.Distance(Node, Data.iloc[i,:])
                
        #keep only the k lowest distances using the set K value
        return NeighbourList.sort_values(by='Distance', ascending=True).iloc[0 : self.K, :] 

    def ClassifyNewNode(self, NearestNeighbours: pd.DataFrame, node):
        Grades: list = NearestNeighbours.iloc[:,NearestNeighbours.shape[1]-2].tolist() #Get a list of all grades (duplicates included)    
        GradeCount: dict = {i: Grades.count(i) for i in Grades} #Count all occurences in the GradeList, storing them in a dictionary

        newnode = node #new node as not to alter the actual dataset
        newnode['Distance'] = sorted(GradeCount)[0] #set value to most occurring grade found in the neighbours

        return newnode

    def KNN(self, TrainData, TestData, K):
        Results = TestData
        for i in range(0, TestData.shape[0]):
            NN = self.FindNearestNeighbours(TrainData, TestData.iloc[i,:])
            Results.iloc[i,:] = self.ClassifyNewNode(NN, TestData.iloc[i,:-1])

        return Results


    def Distance(self, node1, node2) -> float:
        #for all parameters in the node, square the distance between node1 and node2, sum the distance values together
        x: float = sum([(node1[i] - node2[i]) ** 2 for i in range((len(node1) - 1))])

        return math.sqrt(x)