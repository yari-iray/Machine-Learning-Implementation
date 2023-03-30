import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras


DataSplit: dict = {"training": 0.6, "validation": 0.2, "test": 0.2 }
K: int = 5
Dataset: str = "milknew.csv"
np.random.seed(1)

class DataFunctions:
    def LoadDataset():
        CsvData = pd.read_csv("Machine-Learning-Implementation/Machine Learning Project/milknew.csv") #
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
    def __init__(self, Data: np.array, NetworkSize: list):
        self.Input = NeuralNetwork.GetInputmatrix(Data)
        self.ExpectedOutput = NeuralNetwork.FormatTargetValues(Data)
        self.NetworkSize = NetworkSize
        self.Weights = NeuralNetwork.GetStartWeights(self)
        self.Neurons, self.y = NeuralNetwork.ComputeNeuronNetwork(self, 0)

    def GetInputmatrix(Data) -> np.array:
        X = Data.iloc[:, :-1].values # get all the rows and all the columns except the last one 
        X = np.array(X)  #change the matrix into a numpy array (this will make it easier and faster to work with) 

        return X

    def FormatTargetValues(Data) -> np.array:
        Y: np.ndarray = np.array(Data.iloc[:, -1].values) #change the matrix into a numpy array
        
        for val in Y:
            match val:
                case "low": val = 0
                case "medium": val = 1
                case "high": val = 2  

        return Y
    
    def Sigmoid(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))
    
    def SigmoidDerivative(x: np.array) -> np.array:
        return NeuralNetwork.Sigmoid(x) * (1 - NeuralNetwork.Sigmoid(x))
    

    def GetLoss(self, i: int) -> float:
        return np.square(self.y - self.ExpectedOutput[i])

    def GetLossDerivative(self, i: int) -> float:
        return 2 * (self.ExpectedOutput[i] - self.y)


    def GetLayerStartWeights(InputLayerSize: int, OutputLayerSize: int):
        #Create a matrix of random weights between -1 and 1
        #This matrix will have the same amount of columns as the input, and the same amount of rows as the output
        Weights = np.random.rand(OutputLayerSize, InputLayerSize)
        Weights = Weights * 2 - 1

        return Weights

    def GetStartWeights(self):
        # Inititialze the starting weights for the input layer of the network as an empty list
        # Fill that empty list with the weights for the input layer first.
        TotalWeights: list = []
        TotalWeights.append(NeuralNetwork.GetLayerStartWeights(self.Input.shape[1], self.NetworkSize[0]))

        # Iteratively fill the weights list with the weights for the hidden layers
        for i in range(1,len(self.NetworkSize)):
            Weights: np.array = NeuralNetwork.GetLayerStartWeights(self.NetworkSize[i-1], self.NetworkSize[i])
            TotalWeights.append(Weights)

        # Add the weights for the output layer
        TotalWeights.append(NeuralNetwork.GetLayerStartWeights(self.NetworkSize[-1], 1))

        return TotalWeights


    def ComputeNeuronNetwork(self, i: int):
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
                CurrentLayer[:,n] = NeuralNetwork.Sigmoid(np.dot(PreviousLayer, self.Weights[i][n,:]))

            AllNeurons.append(CurrentLayer)
            PreviousLayer = CurrentLayer.copy()

        # Compute the output of the network Using the final hidden layer
        y = NeuralNetwork.Sigmoid(np.dot(CurrentLayer, self.Weights[-1][0,:]))


        return AllNeurons, y

    def BackPropagate(self, i: int):
        # Define the new weights as a copy of the old weights
        # Compute the derivative of the loss function
        NewWeights = self.Weights.copy()
        dy = NeuralNetwork.GetLossDerivative(self, i)

        # Compute the change in weights for the output layer
        NewWeights[-1][0,:] = NewWeights[-1][0,:] - self.Neurons[-1] * dy

        # Compute the change in values for the output layer
        dValues = np.zeros((1, self.NetworkSize[-1]))
        dValues = NeuralNetwork.SigmoidDerivative(self.Neurons[-1]) * dy


        # For each layer, compute the change in values for the current layer for both the change in neuron value (to compute the new weights) and the change in weights
        for i in reversed(range(len(self.NetworkSize))):

            # Compute the change in weights for the current layer, for every node separately
            for n in range(self.NetworkSize[i]):
                NewWeights[i][n,:] = NewWeights[i][n,:] - self.Neurons[i] * dValues[0,n]

            # Reset the change in values to the size of the next relevant layer
            dValues = np.zeros((1, self.NetworkSize[i]))
            dValues = NeuralNetwork.SigmoidDerivative(self.Neurons[i]) * dy

        self.Weights = NewWeights.copy() # Set the new weights as the useable weights


    def TrainNetwork():
        pass

    def TestNetwork(self, testData: pd.DataFrame):
        testSet = self.PrepareInput(testData)

        expectedOutput = self.ClassesToNumericValues(testData)

        outputValues = self.PredictValue(testSet)
        outputValues = self.GetClassificationByNumericPrediction(outputValues)
        
        errors = sum([1 for i in range(len(outputValues)) if outputValues[i] != expectedOutput[i]])

        return errors

    def GetClassificationByNumericPrediction(self, predictedValues: list) -> list:
        numberClassPairs = {0: "low", 1: "medium", 1: "high"}

        # restrict range of values to prevent lookupErrors
        for a in predictedValues:
            if a > 3:
                a = 3
            elif a < 0: 
                a = 0

        return [numberClassPairs[round(predictedValue)] for predictedValue in predictedValues]

    def PredictValue(self, data: np.ndarray):
        result = []

        for val in data:
            result.append(np.dot(val, self.Weights[-1]))

        return result


class NEWralNetwork:
    def __init__(self, TrainData: pd.DataFrame, ValidationData: pd.DataFrame):
        self.TrainData = TrainData
        self.ValidationData = ValidationData

    def run(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10,input_shape=(7,), activation='sigmoid'),
            tf.keras.layers.Dense(10, activation='sigmoid'),
            tf.keras.layers.Dense(3, activation='softmax')
            ])

        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
        
        model.fit(
            self.TrainData.iloc[1:,:-1].values, 
            self.TrainData.iloc[1:,-1].values, 
            batch_size=len(self.TrainData)
            )

def Main():
    TrainData, ValidationData, TestData = DataFunctions.LoadDataset()
    TrainData = DataFunctions.NormalizeData(TrainData)
    ValidationData = DataFunctions.NormalizeData(ValidationData)
    TestData = DataFunctions.NormalizeData(TestData)

    #create a neural network instance with 2 hidden layers of 1 neuron each
    #Network = NeuralNetwork(TrainData, [7,3,4,1,8,4])


    #print('old weights')
    #print(Network.Weights)
    
    Network = NEWralNetwork(TrainData, ValidationData)
    Network.run()

if __name__ == "__main__":
    Main()