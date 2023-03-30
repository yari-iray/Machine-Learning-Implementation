from dataclasses import dataclass
import numpy as np
import pandas as pd
import math

class NeuralNetwork:
    def __init__(self, data: np.array, networkSize: list, learningRate: int):
        """
        output
        input
        neurons
        weights
        expected output
        learning rate
        """
        self.Input = self.PrepareInput(data)
        self.ExpectedOutput = self.ClassesToNumericValues(data)
        self.NetworkSize = networkSize
        self.GetStartWeights()
        self.LearningRate = learningRate

        self.ComputeNeuralNetwork(0)

    def PrepareInput(self, data) -> np.ndarray:
        X = data.iloc[:, :-1].values # get all the rows and all the columns except the last one 
        X = np.array(X)  #change the matrix into a numpy array (this will make it easier and faster to work with) 

        return X

    def ClassesToNumericValues(self, data) -> np.array:
        #convert classes into grade numbers
        grades: np.ndarray = np.array(data.iloc[:, -1].values)
        
        gradeNumberPairs = {'low': 0, 'medium': 0.5, 'high': 1}
        for i in range(len(grades)):
            grades[i] = gradeNumberPairs[grades[i]]
        
        return grades
    
    def Sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def SigmoidDerivative(self, x: np.ndarray) -> np.array:
        sig_x = self.Sigmoid(x)
        return sig_x * (1 - sig_x)    

    def GetLoss(self, i: int) -> float:
        return np.square(self.Output - self.ExpectedOutput[i])

    def GetLossDerivative(self, i: int) -> float:
        return 2 * (self.ExpectedOutput[i] - self.Output)
    
    def GetLayerStartWeights(self, InputLayerSize: int, OutputLayerSize: int):
        return np.random.rand(InputLayerSize, OutputLayerSize)

    def GetStartWeights(self):
        Weights = []
        for i in range(len(self.NetworkSize)):
            if i == 0:
                Weights.append(self.GetLayerStartWeights(self.Input.shape[1], self.NetworkSize[i]))
            else:
                Weights.append(self.GetLayerStartWeights(self.NetworkSize[i - 1], self.NetworkSize[i]))
        
        self.Weights = Weights

    def ComputeNeuralNetwork(self, row: int):
        neurons = [self.Input[row]]
        for i in range(len(self.Weights)):
            neurons.append(self.Sigmoid(np.dot(neurons[i], self.Weights[i])))
        
        self.Output = neurons[-1]
        self.ExpectedOutput = self.Input[row]
        self.Neurons = neurons


    def BackPropagate(self):
        deltas = []
        for i in range(len(self.Neurons) - 1, -1, -1):
            if i == len(self.Neurons) - 1:
                deltas.append(self.GetLossDerivative(i) * self.SigmoidDerivative(self.Neurons[i]))
            else:
                deltas.append(np.dot(deltas[-1], self.Weights[i].T) * self.SigmoidDerivative(self.Neurons[i]))
        
        deltas.reverse()
        for i in range(len(self.Weights)):
            self.Weights[i] += self.LearningRate * self.Neurons[i].T.dot(deltas[i])


    def TrainNetwork(self):
        for i in range (1000):
            for i in range(len(self.Input)):
                self.ComputeNeuralNetwork(i)
                self.BackPropagate()

    def TestNetwork(self, testData: pd.DataFrame):
        ExpectedOutput = self.ClassesToNumericValues(testData)
        Input = self.PrepareInput(testData)
        PredictedOutput = np.array([])
        for i in range(len(Input)):
            self.ComputeNeuralNetwork(i)
            PredictedOutput = np.append(PredictedOutput, self.Output)
    
        return PredictedOutput, ExpectedOutput

    def GetClassificationByNumericPrediction(self, predictedValues: list) -> list:
        pass