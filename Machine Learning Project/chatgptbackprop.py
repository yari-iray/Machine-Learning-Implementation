import numpy as np

class chatgpt:


    def BackPropagateGPTTheFirstOneDoNotUse(self, index: int):
        ## Define the new weights as a copy of the old weights
        newWeights = self.Weights.copy()
    
        ## Compute the derivative of the loss function
        dy = self.GetLossDerivative(index)
    
        ## Compute the change in weights for the output layer
        newWeights[-1][0,:] = newWeights[-1][0,:] - self.Neurons[-1] * dy * self.LearningRate
    
        # Compute the change in values for the output layer
        dValues = np.zeros((1, self.NetworkSize[-1]))
        dValues[0,:] = self.SigmoidDerivative(np.dot(self.Neurons[-2], self.Weights[-1].T)) * dy
    
        # For each layer, compute the change in values for the current layer for both the change in neuron value (to compute the new weights) and the change in weights
        for i in reversed(range(1, len(self.NetworkSize))):
            # Compute the change in weights for the current layer
            newWeights[i] -= self.LearningRate * np.outer(dValues, self.Neurons[i-1])
        
            # Compute the change in values for the current layer
            dValues = np.zeros((1, self.NetworkSize[i]))
            dValues[0,:] = self.SigmoidDerivative(np.dot(self.Neurons[i-1], self.Weights[i].T)) * np.dot(dValues, self.Weights[i])
        
            # Update the previous layer
            previousLayer = self.Neurons[i-1]
            # Update the weights for the current layer
            newWeights[i-1] -= self.LearningRate * np.outer(dValues, previousLayer)
    
        # Update the weights for the input layer
        newWeights[0] -= self.LearningRate * np.outer(dValues, self.Input[index])
    
        # Set the new weights as the current weights
        self.Weights = newWeights

    def BackPropBackup(self, index: int):
        # Define the new weights as a copy of the old weights
        # Compute the derivative of the loss function
        newWeights = self.Weights.copy()
        dy = self.GetLossDerivative(index)

        # Compute the change in weights for the output layer
        newWeights[-1][0,:] = newWeights[-1][0,:] - self.Neurons[-1] * dy

        # Compute the change in values for the output layer
        dValues = np.zeros((1, self.NetworkSize[-1]))
        dValues = self.SigmoidDerivative(self.Weights[-1]) * dy

        # For each layer, compute the change in values for the current layer for both the change in neuron value (to compute the new weights) and the change in weights
        for i in reversed(range(len(self.NetworkSize))):

            # Compute the change in weights for the current layer, for every node separately
            for n in range(self.NetworkSize[i]):
                newWeights[i][n,:] = newWeights[i][n,:] - self.Neurons[i] * dValues[0,n] 
                

            # Reset the change in values to the size of the next relevant layer
            dValues = np.zeros((1, self.NetworkSize[i])) # is this line redundant?
            dValues = self.SigmoidDerivative(self.Neurons[i]) * dy

        self.Weights = newWeights.copy() # Set the new weights as the useable weights