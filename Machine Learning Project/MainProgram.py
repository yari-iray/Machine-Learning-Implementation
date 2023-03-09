import math
import pandas as pd

DataSplit: dict = {"training": 0.6, "validation": 0.2, "test": 0.2 }
K: int = 5
Dataset: str = "milknew.csv"

def FindNearestNeighbours(DataFrame, Node):
    # find and return k nearest neighbours to node by ordering list and taking the k nearest ones
    NeighbourList = DataFrame #Create a separate Dataframe to store the distance to the given Node
    NeighbourList['Distance'] = math.inf #Initialize as infinity

    #Find all the distances to the points in the dataframe from Node.
    for i in range(0, DataFrame.shape[0]):
        NeighbourList.iloc[i,NeighbourList.shape[1]-1] = Distance(Node, DataFrame.iloc[i,:])
    
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
        NN = FindNearestNeighbours(TrainData, TestData.iloc[i,:])
        Results.iloc[i,:] = ClassifyNewNode(NN, TestData.iloc[i,:-1])

    return Results


def Distance(node1, node2) -> float:
    #for all parameters in the node, square the distance between node1 and node2, sum the distance values together
    x = sum([(node1[i] - node2[i]) ** 2 for i in range((len(node1) - 1))])
    return math.sqrt(x)


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

def Main():
    TrainData, ValidationData, TestData = DataFunctions.LoadDataset()
    TrainData = DataFunctions.NormalizeData(TrainData)
    ValidationData = DataFunctions.NormalizeData(ValidationData)
    TestData = DataFunctions.NormalizeData(TestData)

    print("Test data")
    print(TestData)

    print("KNN classification")
    print(KNN(TrainData, TestData))

if __name__ == "__main__":
    Main()