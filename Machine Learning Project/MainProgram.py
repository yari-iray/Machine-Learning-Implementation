import math
import pandas as pd
import numpy as np


DataSplit: dict = {"training": 0.6, "validation": 0.2, "test": 0.2 }
K: int = 5
Dataset: str = "Machine Learning Project/milknew.csv"
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

class NeuralNetwork:
    def Sigmoid(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))
    

    def SigmoidDerivative(x: np.array) -> np.array:
        return NeuralNetwork.Sigmoid(x) * (1 - NeuralNetwork.Sigmoid(x))
    

    def GetLoss(y: np.array, Targets: np.array) -> np.array:
        return np.sum((y - Targets) ** 2)
    

    def GetLossDerivative(y: np.array, Targets: np.array) -> np.array:
        return 2 * (y - Targets)

    def softMax(data):
        exp_data = [np.exp(i - np.max(data)) for i in x] # calculate explonentials of all the data
        sum_exp_data = sum(exp_data) # summing all the exponentials
        softmax_data = [i / sum_exp_data for i in exp_data] #getting an array of all the probabilities.

        return max(softmax_data)


    def GetInputmatrix(Input) -> np.array:
        X = Input.iloc[:, :-1].values # get all the rows and all the columns except the last one 
        X = np.array(X)  #change the matrix into a numpy array (this will make it easier and faster to work with) 

        return X


    def GetTargetValues(Input) -> np.array:
        Y = Input.iloc[:, -1].values #get the last(target) row
        Y = np.array(Y) #change the matrix into a numpy array

        #Change the target values into numbers, so they can be used with the network

        return Y


    def GetLayerStartWeights(InputLayerSize: int, OutputLayerSize: int):
        #Create a matrix of random weights between -1 and 1
        #This matrix will have the same amount of columns as the input, and the same amount of rows as the output
        Weights = np.random.rand(OutputLayerSize, InputLayerSize)
        Weights = Weights * 2 - 1

        return Weights
    

    def GetStartWeights(Input: np.array, NetworkSize: list):
        # Inititialze the starting weights for the input layer of the network as an empty list
        # Fill that empty list with the weights for the input layer first.
        TotalWeights: list = []
        TotalWeights.append(NeuralNetwork.GetLayerStartWeights(Input.shape[1], NetworkSize[0]))

        # Iteratively fill the weights list with the weights for the hidden layers
        for i in range(1,len(NetworkSize)):
            Weights: np.array = NeuralNetwork.GetLayerStartWeights(NetworkSize[i-1], NetworkSize[i])
            TotalWeights.append(Weights)

        # Add the weights for the output layer
        TotalWeights.append(NeuralNetwork.GetLayerStartWeights(NetworkSize[-1], 1))

        return TotalWeights


    def ComputeNeuronNetwork(NetworkSize: list, Input: np.array, Weights: list):
        # Define the first (previous) layer as the input layer
        # Define the current layer as an empty array of the size of the first hidden layer
        PreviousLayer: np.array = Input
        CurrentLayer: np.array = np.zeros((Input.shape[0], NetworkSize[0]))

        # For each layer, for each neuron compute the output of the neuron and store it in the current layer
        # Then, set the current layer as the previous layer and repeat the process for the next layer
        for i in range(len(NetworkSize)):
            for n in range(NetworkSize[i]):
                CurrentLayer[:,n] = NeuralNetwork.Sigmoid(np.dot(PreviousLayer, Weights[i][n,:]))

            PreviousLayer = CurrentLayer.copy()

        # Compute the output of the network Using the final hidden layer
        y = NeuralNetwork.Sigmoid(np.dot(CurrentLayer, Weights[-1][0,:]))

        return y


        def BackpropagateNetwork():
            pass


        def TrainNetwork():
            pass


def Main():
    TrainData, ValidationData, TestData = DataFunctions.LoadDataset()
    TrainData = DataFunctions.NormalizeData(TrainData)
    ValidationData = DataFunctions.NormalizeData(ValidationData)
    TestData = DataFunctions.NormalizeData(TestData)

    inputmatrix = NeuralNetwork.GetInputmatrix(TrainData)
    print("Input matrix")
    print(inputmatrix)
    print("Starting Weights")
    print(NeuralNetwork.GetStartWeights(inputmatrix, [3, 4, 2, 2]))

    print("Computed value")
    print(NeuralNetwork.ComputeNeuronNetwork([2,2], inputmatrix, NeuralNetwork.GetStartWeights(inputmatrix, [2,2])))
    

if __name__ == "__main__":
    Main()