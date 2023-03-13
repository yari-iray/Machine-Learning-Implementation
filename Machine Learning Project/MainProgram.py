from math import sqrt, exp
#from operator import indexOf
#from re import M
import numpy as np
#import matplotlib as mlt
import pandas as pd

DataSplit: dict = {"training": 0.6, "validation": 0.2, "test": 0.2 }
K: int = 5 #temporary value, just so we have one right now
PathToDataset: str = "Machine Learning Project/milknew.csv"

class DataFunctions:
    def LoadDataset():
        CsvData = pd.read_csv(PathToDataset)
        Length: int = len(CsvData) #Needed to make a good split of data

        #The data has to be split into training, validation and test data using predefined percentages.
        TrainData = CsvData[0: int(Length * DataSplit["training"])] 
        ValidationData = CsvData[int(Length * DataSplit["training"]): int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"])]
        TestData = CsvData[int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"]): Length] 

        return TrainData, ValidationData, TestData


    def NormalizeData(DataSet):
        n: int = len(DataSet.columns) #Find the columns so we know how far to iterate

        for i in range(n-1): #range is n-1 because the last column does not contain numbers, but the class.
            LowerBound: int = min(DataSet.iloc[:,i]) #Find the minimum value in the current column
            UpperBound: int = max(DataSet.iloc[:,i]) #Find the maximum value in the current column

            DataSet.iloc[:,i] = (DataSet.iloc[:,i] - LowerBound) / (UpperBound - LowerBound) #Using this function, the column will be normalized to be between 0 and 1
    
        return DataSet

class KNN:
    def Distance(node1, node2) -> float:
        x = sum( [(node1[i] - node2[i]) ** 2 for i in range((len(node1) - 1))] )
        return sqrt(x)
    
    def FindNearestNeighbours(DataFrame, Node, k: int):
        # find and return k nearest neighbours to node
        # order the list and take k neares ones
        NeighbourList = DataFrame.copy #Create a separate Dataframe to store the distance to the given Node
        NeighbourList['Distance'] = 99999 #Set the distance to a very high number at first.

        #Find all the distances to the points in the dataframe from Node.
        for i in range(0, DataFrame.shape[0]):
            NeighbourList.iloc[i,NeighbourList.shape[1]-1] = KNN.Distance(Node, DataFrame.iloc[i,:])
        
        NeighbourList = NeighbourList.sort_values(by='Distance', ascending=True) #Sort the dataframe by distance
        NeighbourList = NeighbourList.iloc[0:k,:] #Keep only the k lowest distances, and return them.
        
        return NeighbourList

    def ClassifyNewNode(NearNeighbours, node):
        GradeList: list = NearNeighbours.iloc[:,NearNeighbours.shape[1]-2].tolist() #Get a list of all grades (duplicates included)    
        GradeCount: dict = {i: GradeList.count(i) for i in GradeList} #Count all occurences in the GradeList, storing them in a dictionary
        
        newnode = node #new node as not to alter the actual dataset
        newnode['Distance'] = sorted(GradeCount)[0] #set the value of the node to the most common Grade found in the Neighbours

        return newnode

    def KNN(TrainData, TestData, k: int):
        Results = TestData.copy() #Create a copy of the TestData dataframe to store the results in
        for i in range(0, TestData.shape[0]):
            NN = KNN.FindNearestNeighbours(TrainData, TestData.iloc[i,:], k)
            Results.iloc[i,:] = KNN.ClassifyNewNode(NN, TestData.iloc[i,:-1])

        return Results

class NeuralNetwork:
    def Sigmoid(x: float):
        return 1 / (1 + exp(-x))

    def Relu(x:float): #ReLU activation function
        return np.maximum(0, x)

    def softMax(data):
        exp_data = [np.exp(i - np.max(data)) for i in x]

        sum_exp_data = sum(exp_data)

        softmax_data = [i / sum_exp_data for i in exp_data]

        return max(softmax_data)
    
    def GetInputmatrix(Input):
        X = Input.iloc[:, :-1].values # get all the rows and all the columns except the last one 
        X = np.array(X)  #change the matrix into a numpy array (this will make it easier and faster to work with) 

        return X

    def GetTargetValues(Input):
        Y = Input.iloc[:, -1].values #get the last(target) row
        Y = np.array(Y) #change the matrid into a numpy array

        return Y

    def GetStartWeights(Input: np.array):
        #Create a matrix of random weights of size (number of features, 1)
        #The weights are initialized to a random value between -1 and 1
        Weights: np.array = np.random.rand(Input.shape[1]-1, 1)
        Weights = Weights * 2 - 1

        return Weights
    
    def ComputeNeuron(Input: np.array, Weights: np.array):
        #Make an empty array and variable, to later store the actual values of the neuron
        Neuron: np.array = np.zeros((Input.shape[0], 1))
        value: float = 0
        
        #Iterate over all the rows in the input matrix (cannot do the entire matrix at once, the sigmoid does not allow for that)
        #Compute the value of a neuron using the sigmoid function and the element wise multiplication of the input and the weights
        for i in range(Input.shape[0]): 
            value = NeuralNetwork.Sigmoid(np.dot(Input[i,:], Weights)) 
            Neuron[i,0] = value

        return Neuron



def Main():
    TrainData, ValidationData, TestData = DataFunctions.LoadDataset()
    TrainData = DataFunctions.NormalizeData(TrainData)
    ValidationData = DataFunctions.NormalizeData(ValidationData)
    TestData = DataFunctions.NormalizeData(TestData)

    print(NeuralNetwork.ComputeNeuron(NeuralNetwork.GetInputmatrix(TrainData), NeuralNetwork.GetStartWeights(TrainData)))

if __name__ == "__main__":
    Main()
