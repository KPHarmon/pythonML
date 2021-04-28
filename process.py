import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class Model:
    

class Cluster:
    nodes = []

    def add_nodes(self, list_of_nodes):
        for node in list_of_nodes:
            self.nodes.append(node)

    def remove_nodes(self):
        temp_list = self.nodes
        self.nodes = []
        return temp_list

    def read_cluster():
        for node in self.nodes:
            print("Node: \n", node)

class Node:
    def __init__(self, key, features, values):
        self.data = {}
        self.key = key
       
        for index, feature in enumerate(features):
            self.data[feature] = values[index]

    def __repr__(self):
        string = "Key: " + str(self.key) + "\n"
        for key in self.data:
            string = string + "\t" + str(key) + " -> " + str(self.data[key]) + "\n"
        return string

def process_csv(filename_csv):
   
    # Iris dataset is from sklearn
    if filename_csv == "iris":
        iris = load_iris()
        dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
    else:
        dataframe = pd.read_csv(filename_csv)
        
    return dataframe

# Create a list of nodes from the dataframe
def initialize_data(dataframe):
    features = []
    nodes = []

    # Get Data Columns
    for feature in dataframe.columns:
        features.append(feature)

    # Get Data Values
    for item in dataframe.iterrows():
        key = item[0]
        values = []

        for i in range(len(dataframe.columns)):
            values.append(item[1][i])

        # Represent data as a node
        new_node = Node(key, features, values)
        nodes.append(new_node)

    return nodes
