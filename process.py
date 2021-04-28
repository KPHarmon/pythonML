import sys
import math
import random
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

class Model:
    def __init__(self, dataframe, algorithm, k):
        self.algorithm = algorithm
        self.dataframe = dataframe
        self.model = []
        self.k = k

    def __repr__(self):
        return self.model

    def run(self):
        #if algorithm.upper() == "SINGLE":

        if self.algorithm.upper() == "AVERAGE":
            self.average_algorithm()
            return(self.model)
      
        elif self.algorithm.upper() == "LLOYD":
            self.lloyd_algorithm()
            return self.model
        
        else:
            print("Unsupported Algorithm")

    # Lloyd's Algorithm
    def lloyd_algorithm(self):
        nodes = initialize_data(self.dataframe)
        
        old_clustering = [0]
        new_clustering = [1]

        first_iteration = 1
        while old_clustering != new_clustering:
            old_clustering = new_clustering

            # Choose k random centers
            if first_iteration == 1:
                first_iteration = 0
                centers = []
                for integer in range(self.k):
                    centers.append(random.choice(nodes))
            
            else:
                centers = []
                for cluster in old_clustering:
                    centers.append(cluster.average())
        
            # Find the distance between each node and the center
            distances = []
            for node1 in nodes:
                shortest_distance = Shortest_Distance()
                for center in centers:
                    if distance(node1, center) < shortest_distance.distance and distance(node1, center) != 0:
                        shortest_distance.update(distance(node1, center), [node1], [center])
                distances.append(shortest_distance)
            
            # Create clusters
            new_clustering = []
            for center in centers:
                temp_cluster = []
                for distance in distances:
                    if center == distance.cluster2:
                        temp_cluster.append(cluster1)
                new_clustering.append(temp_cluster)


    # Average Linkage Algorithm
    def average_algorithm(self):

        # Initialize Singletons
        nodes = initialize_data(self.dataframe)
        for node in nodes:
            self.model.append([node])

        # Perform Algorithm
        while len(self.model) > self.k:

            shortest_distance = Shortest_Distance()
            for i, cluster1 in enumerate(self.model[:-1]):
                for cluster2 in self.model[i+1:]:

                    if distance(cluster1.average(), cluster2.average()) < shortest_distance.distance:
                        shortest_distance.update(distance(cluster1.average(), cluster2.average()), cluster1, cluster2) 
                
            # Merge Clusters
            self.model.remove(shortest_distance.cluster1)
            self.model.remove(shortest_distance.cluster2)
            self.model.append(Cluster(shortest_distance.cluster1.nodes + shortest_distance.cluster2.nodes))

class Shortest_Distance():
    def __init__(self):
        self.distance = 9999999

    def update(self, distance, cluster1, cluster2):
        self.distance = distance
        self.cluster1 = cluster1
        self.cluster2 = cluster2
        
class Cluster:

    # Initialize with a list of nodes
    def __init__(self, list_of_nodes):
        self.nodes = []
        for node in list_of_nodes:
            self.nodes.append(node)

    # Add a list of nodes
    def add_nodes(self, list_of_nodes):
        for node in list_of_nodes:
            self.nodes.append(node)

    # Remove all of the nodes from a cluster, returns the deleted nodes
    def remove_nodes(self):
        temp_list = self.nodes
        self.nodes = []
        return temp_list

    # Function to create an Average Node
    def average(self):

        features = []
        average_values = []
        temp_id = ""

        # Use the first node as a template
        for key in self.nodes[0].data:
            features.append(key)

        for feature in features:
            average = 0
            for node in self.nodes:
                average += node.data[feature]
                temp_id += str(node.key)
            average_values.append(average/len(self.nodes))

        return Node(temp_id, features, average_values)

    # Print out the cluster
    def __repr__(self):
        return str(self.nodes)
        #string = "["
        #for node in self.nodes:
        #    string = string + str(node) + ", "
        #return string + "]\n"

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

# Helper function to find the distance between two nodes
def distance(node1, node2):
    x = 0
    for key in node1.data:
        x = x + (node1.data[key] - node2.data[key])**2
        
    return math.sqrt(x)
    
def Hamming(model1, model2):
    
