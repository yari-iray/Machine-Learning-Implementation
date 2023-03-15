import math
import pandas as pd
import numpy as np


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
    def __init__(self, data: np.array, networkSize: list):
        self.Input = self.PrepareInput(data)
        self.ExpectedOutput = self.FormatTargetValues(data)
        self.NetworkSize = networkSize
        self.Weights = self.GetStartWeights()
        self.Neurons, self.Output = self.ComputeNeuralNetwork(0)

    def PrepareInput(self, data) -> np.array:
        X = data.iloc[:, :-1].values # get all the rows and all the columns except the last one 
        X = np.array(X)  #change the matrix into a numpy array (this will make it easier and faster to work with) 

        return X

    def FormatTargetValues(self, data) -> np.array:
        #convert classes into grades
        grades: np.ndarray = np.array(data.iloc[:, -1].values)
        
        gradeNumberPairs = {"low": 0, "medium": 1, "high": 2}
        for val in gradeNumberPairs:
            val = grades[val]

        return grades

    def GetStartWeights(self):
        # Inititialze the starting weights for the input layer of the network as an empty list
        # Fill that empty list with the weights for the input layer first.
        TotalWeights: list = []
        TotalWeights.append(self.GetLayerStartWeights(self.Input.shape[1], self.NetworkSize[0]))

        # Iteratively fill the weights list with the weights for the hidden layers
        for i in range(1,len(self.NetworkSize)):
            Weights: np.array = self.GetLayerStartWeights(self.NetworkSize[i-1], self.NetworkSize[i])
            TotalWeights.append(Weights)

        # Add the weights for the output layer
        TotalWeights.append(self.GetLayerStartWeights(self.NetworkSize[-1], 1))

        return TotalWeights

    def GetLayerStartWeights(self, InputLayerSize: int, OutputLayerSize: int):
        #Create a matrix of random weights between -1 and 1
        #This matrix will have the same amount of columns as the input, and the same amount of rows as the output
        Weights = np.random.rand(OutputLayerSize, InputLayerSize)
        Weights = Weights * 2 - 1

        return Weights
    
    def Sigmoid(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))
    
    def SigmoidDerivative(self, x: np.array) -> np.array:
        return self.Sigmoid(x) * (1 - self.Sigmoid(x))
    

    def GetLoss(self, i: int) -> float:
        return np.square(self.Output - self.ExpectedOutput[i])

    def GetLossDerivative(self, i: int) -> float:
        return 2 * (self.ExpectedOutput[i] - self.Output)

    def ComputeNeuralNetwork(self, i: int):
        # Define the first (previous) layer as the input layer
        # Define the current layer as an empty array of the size of the first hidden layer
        AllNeurons: list = []
        PreviousLayer: np.array = self.Input[i].copy()
        AllNeurons.append(PreviousLayer)
        CurrentLayer: np.array = np.zeros((1, self.NetworkSize[0]))

        # For each layer, for each neuron compute the output of the neuron and store it in the current layer
        # Then, set the current layer as the previous layer and repeat the process for the next layer
        for i in range(len(self.NetworkSize)):
            CurrentLayer = np.zeros((1, self.NetworkSize[i]))
            for n in range(self.NetworkSize[i]):
                CurrentLayer[:,n] = self.Sigmoid(np.dot(PreviousLayer, self.Weights[i][n,:]))

            AllNeurons.append(CurrentLayer)
            PreviousLayer = CurrentLayer.copy()

        # Compute the output of the network Using the final hidden layer
        output = self.Sigmoid(np.dot(CurrentLayer, self.Weights[-1][0,:]))

        return AllNeurons, output

    def BackPropagate(self, i: int):
        # Define the new weights as a copy of the old weights
        # Compute the derivative of the loss function
        NewWeights = self.Weights.copy()
        dy = self.GetLossDerivative(i)

        # Compute the change in weights for the output layer
        NewWeights[-1][0,:] = NewWeights[-1][0,:] - self.Neurons[-1] * dy

        # Compute the change in values for the output layer
        dValues = np.zeros((1, self.NetworkSize[-1]))
        dValues = self.SigmoidDerivative(self.Neurons[-1]) * dy


        # For each layer, compute the change in values for the current layer for both the change in neuron value (to compute the new weights) and the change in weights
        for i in range(len(self.NetworkSize), 0):

            # Compute the change in weights for the current layer, for every node separately
            for n in range(self.NetworkSize[i]):
                NewWeights[i][n,:] = NewWeights[i][n,:] - self.Neurons[i] * dValues[0,n]

            # Reset the change in values to the size of the next relevant layer
            dValues = np.zeros((1, self.NetworkSize[i])) # is this line redundant?
            dValues = self.SigmoidDerivative(self.Neurons[i]) * dy

        self.Weights = NewWeights.copy() # Set the new weights as the useable weights

    def TrainNetwork(self):
        for i in range(len(self.Input)):
            pass

    def TestNetwork(self, TestData: pd.DataFrame):
        #change input to the input matrix
        self.Input = self.GetInputMatrix(TestData)
        # should not be needed, we need to store the neurons in a good way
        # so we can 
        errorCount = 0

        for i in range(len(self.Output)):
            if self.Output[i] != self.ExpectedOutput[i]:
                errorCount += 1

        return errorCount



def Main():
    TrainData, ValidationData, TestData = DataFunctions.LoadDataset()
    TrainData = DataFunctions.NormalizeData(TrainData)
    ValidationData = DataFunctions.NormalizeData(ValidationData)
    TestData = DataFunctions.NormalizeData(TestData)

    #create a neural network instance with 2 hidden layers of 1 neuron each
    Network = NeuralNetwork(TrainData, [7,3,4,1,8,4])



    print('old weights')
    print(Network.Weights)
    

if __name__ == "__main__":
    Main()