import numpy as np
import pandas as pd

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) #define the sigmoid function

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z)) #derivative of the sigmoid function


class Network:
    def __init__(self, layers):
        self.num_layers = len(layers) #number of layers in the network
        self.layers = layers #list of the number of neurons in each layer
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        #list of weights (in terms of a tuple). x is the neurons in the current layer, y is the neurons in the next layer.
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])] #weights are random in the beginning

    
    def feedforward(self, a):
        for w in self.weights: 
            a = sigmoid(np.dot(w, a))
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data is not None: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            #np.random.shuffle(training_data)
            training_data = training_data.sample(frac=1).reset_index(drop=True)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            
        
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


DataSplit: dict = {"training": 0.6, "validation": 0.2, "test": 0.2 }
Dataset: str = "Machine-Learning-Implementation/Machine Learning Project/milknew.csv"

class DataFunctions:
    @staticmethod
    def LoadDataset():
        CsvData = pd.read_csv(Dataset)
        Length: int = len(CsvData) #needed to split the data into parts

        #Split data into train, validation, test
        TrainData = CsvData[0: int(Length * DataSplit["training"])] 
        ValidationData = CsvData[int(Length * DataSplit["training"]): int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"])]
        TestData = CsvData[int(Length * DataSplit["training"]) + int(Length * DataSplit["validation"]): Length] 

        return TrainData, ValidationData, TestData

    @staticmethod
    def NormalizeData(DataSet: pd.DataFrame):
        n: int = len(DataSet.columns) #Number of columns

        #the last column contains the grade, which doesn't need to be normalized as it is a classification and not a number
        #loop over all columns and normalize them
        for i in range(n - 1): 
            LowerBound: int = min(DataSet.iloc[:,i]) #Column min
            UpperBound: int = max(DataSet.iloc[:,i]) #Column max

            DataSet.iloc[:,i] = (DataSet.iloc[:,i] - LowerBound) / (UpperBound - LowerBound) #Using this function, the column will be normalized to be between 0 and 1
    
        return DataSet

if __name__ == "__main__":
    print("Starting")
    TrainData, ValidationData, TestData = DataFunctions.LoadDataset()
    TrainData = DataFunctions.NormalizeData(TrainData)
    ValidationData = DataFunctions.NormalizeData(ValidationData)
    TestData = DataFunctions.NormalizeData(TestData)

    Network = Network([7, 3, 1])
    Network.SGD(TrainData, 30, 10, 3.0, TestData)

    print("Done")
