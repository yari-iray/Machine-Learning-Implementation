import math
import pandas as pd

class KNN:
    def FindNearestNeighbours(DataFrame, Node):
        # find and return k nearest neighbours to node by ordering list and taking the k nearest ones
        NeighbourList = DataFrame #Create a separate Dataframe to store the distance to the given Node
        NeighbourList['Distance'] = math.inf #Initialize as infinity

        #Find all the distances to the points in the dataframe from Node.
        for i in range(0, DataFrame.shape[0]):
            NeighbourList.iloc[i,NeighbourList.shape[1]-1] = KNN.Distance(Node, DataFrame.iloc[i,:])
    
        NeighbourList = NeighbourList.sort_values(by='Distance', ascending=True) #Sort the dataframe by distance
        NeighbourList = NeighbourList.iloc[0:K,:] #Keep only the K lowest distances, and return them.
    
        return NeighbourList

    def ClassifyNewNode(NearestNeighbours: pd.DataFrame, node):
        Grades: list = NearestNeighbours.iloc[:,NearestNeighbours.shape[1]-2].tolist() #Get a list of all grades (duplicates included)    
        GradeCount: dict = {i: Grades.count(i) for i in Grades} #Count all occurences in the GradeList, storing them in a dictionary

        newnode = node #new node as not to alter the actual dataset
        newnode['Distance'] = sorted(GradeCount)[0] #set value to most occurring grade found in the neighbours

        return newnode

    def KNN(TrainData, TestData):
        Results = TestData
        for i in range(0, TestData.shape[0]):
            NN = KNN.FindNearestNeighbours(TrainData, TestData.iloc[i,:])
            Results.iloc[i,:] = KNN.ClassifyNewNode(NN, TestData.iloc[i,:-1])

        return Results


    def Distance(node1, node2):
        #for all parameters in the node, square the distance between node1 and node2, sum the distance values together
        x = sum([(node1[i] - node2[i]) ** 2 for i in range((len(node1) - 1))])

        return math.sqrt(x)