from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class NeuralNetworkOutput:
    AllNeurons: list
    SigmoidOutput: float
    AbsoluteOutput: float

class NeuralNetwork:
    def __init__(self, data: np.array, networkSize: list):
        self.Input = self.PrepareInput(data)
        self.ExpectedOutput = self.ClassesToNumericValues(data)
        self.NetworkSize = networkSize
        self.Weights = self.GetStartWeights()

        result = self.ComputeNeuralNetwork(0)

        self.Neurons = result.AllNeurons
        self.Output = result.SigmoidOutput

    def PrepareInput(self, data) -> np.ndarray:
        X = data.iloc[:, :-1].values # get all the rows and all the columns except the last one 
        X = np.array(X)  #change the matrix into a numpy array (this will make it easier and faster to work with) 

        return X

    def ClassesToNumericValues(self, data) -> np.array:
        #convert classes into grade numbers
        grades: np.ndarray = np.array(data.iloc[:, -1].values)
        
        gradeNumberPairs = {'low': 0, 'medium': 1, 'high': 2}
        for i in range(len(grades)):
            grades[i] = gradeNumberPairs[grades[i]]
        
        return grades

    def GetStartWeights(self):
        # Inititialze the starting weights for the input layer of the network as an empty list
        # Fill that empty list with the weights for the input layer first.
        totalWeights: list = []
        totalWeights.append(self.GetLayerStartWeights(self.Input.shape[1], self.NetworkSize[0]))

        # Iteratively fill the weights list with the weights for the hidden layers
        for i in range(1,len(self.NetworkSize)):
            weights: np.array = self.GetLayerStartWeights(self.NetworkSize[i-1], self.NetworkSize[i])
            totalWeights.append(weights)

        # Add the weights for the output layer
        totalWeights.append(self.GetLayerStartWeights(self.NetworkSize[-1], 1))

        return totalWeights

    def GetLayerStartWeights(self, InputLayerSize: int, OutputLayerSize: int):
        #Create a matrix of random weights between -1 and 1
        #This matrix will have the same amount of columns as the input, and the same amount of rows as the output
        weights = np.random.rand(OutputLayerSize, InputLayerSize)

        return weights * 2 - 1
    
    def Sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def SigmoidDerivative(self, x: np.ndarray) -> np.array:
        sig_x = self.Sigmoid(x)
        return sig_x * (1 - sig_x)
    

    def GetLoss(self, i: int) -> float:
        return np.square(self.Output - self.ExpectedOutput[i])

    def GetLossDerivative(self, i: int) -> float:
        return 2 * (self.ExpectedOutput[i] - self.Output[0])

    def ComputeNeuralNetwork(self, index: int) -> NeuralNetworkOutput:
        # Define the first (previous) layer as the input layer
        # Define the current layer as an empty array of the size of the first hidden layer
        previousLayer: np.array = self.Input[index].copy()
        allNeurons = [previousLayer]
        currentLayer: np.array = np.zeros((1, self.NetworkSize[0]))

        # For each layer, for each neuron compute the output of the neuron and store it in the current layer
        # Then, set the current layer as the previous layer and repeat the process for the next layer
        for i in range(len(self.NetworkSize)):
            currentLayer = np.zeros((1, self.NetworkSize[i]))
            for n in range(self.NetworkSize[i]):
                currentLayer[:,n] = self.Sigmoid(np.dot(previousLayer, self.Weights[i][n,:]))

            allNeurons.append(currentLayer)
            previousLayer = currentLayer.copy()

        # Compute the output of the network Using the final hidden layer
        finalLayerDotProduct = np.dot(currentLayer, self.Weights[-1][0,:])
        output = self.Sigmoid(finalLayerDotProduct)

        return NeuralNetworkOutput(allNeurons, output[0], finalLayerDotProduct)

    def BackPropagate(self, index: int):
        # Define the new weights as a copy of the old weights
        # Compute the derivative of the loss function
        newWeights = self.Weights.copy()
        dy = self.GetLossDerivative(index)

        # Compute the change in weights for the output layer
        newWeights[-1][0,:] = newWeights[-1][0,:] - self.Neurons[-1] * dy

        # Compute the change in values for the output layer
        dValues = np.zeros((1, self.NetworkSize[-1]))
        dValues = self.SigmoidDerivative(self.Neurons[-1]) * dy

        # For each layer, compute the change in values for the current layer for both the change in neuron value (to compute the new weights) and the change in weights
        for i in reversed(range(len(self.NetworkSize))):

            # Compute the change in weights for the current layer, for every node separately
            for n in range(self.NetworkSize[i]):
                newWeights[i][n,:] = newWeights[i][n,:] - self.Neurons[i] * dValues[0,n]

            # Reset the change in values to the size of the next relevant layer
            dValues = np.zeros((1, self.NetworkSize[i])) # is this line redundant?
            dValues = self.SigmoidDerivative(self.Neurons[i]) * dy

        self.Weights = newWeights.copy() # Set the new weights as the useable weights

        return newWeights

    def TrainNetwork(self):
        for i in range(len(self.Input)):
            result = self.ComputeNeuralNetwork(i)
            self.Output = result.AbsoluteOutput

            self.BackPropagate(i)

    def TestNetwork(self, testData: pd.DataFrame):
        self.TrainNetwork()

        testSet: np.ndarray = self.PrepareInput(testData)
        expectedClassifications = testData.iloc[:, -1].values

        self.Input = testSet
        self.ExpectedOutput = self.ClassesToNumericValues(testData)

        result = self.PredictValues(testSet)
        outputValues = self.GetClassificationByNumericPrediction(result)

        errors = 0
        for i in range(len(outputValues)):
            if expectedClassifications[i] != outputValues[i]:
                errors += 1

        return errors

    def GetClassificationByNumericPrediction(self, predictedValues: list) -> list:
        numberClassPairs = {0: "low", 1: "medium", 2: "high"}

        # restrict range of values to prevent lookupErrors
        for i in range(len(predictedValues)):
            if predictedValues[i] > 3:
                predictedValues[i] = 3
            elif predictedValues[i] < 0: 
                predictedValues[i] = 0

        return [numberClassPairs[round(value)] for value in predictedValues]

    def PredictValues(self, data: np.ndarray):        
        result = []
        for i in range(len(data)):            
            values = self.ComputeNeuralNetwork(i)

            result.append(values.AbsoluteOutput[0])

        return result