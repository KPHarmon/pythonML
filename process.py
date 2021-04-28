import sys
import math
import random
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.datasets import load_iris

class Model:
    def __init__(self, dataframe, algorithm, k):
        self.algorithm = algorithm
        self.dataframe = dataframe
        self.model = []
        self.k = k

    def __repr__(self):
        return self.model

    def run(self, iterations=0):
        if self.algorithm.upper() == "SINGLE":
            self.single_algorithm()
            return(self.model)
        elif self.algorithm.upper() == "AVERAGE":
            self.average_algorithm()
        elif self.algorithm.upper() == "LLOYD":
            self.lloyd_algorithm()
            return self.model
        
        else:
            print("Unsupported Algorithm")

    # Lloyd's Algorithm
    def lloyd_algorithm(self):

        X = self.dataframe
        k = self.k
        max_iterations = 100
        '''
        X: multidimensional data
        k: number of clusters
        max_iterations: number of repetitions before clusters are established
        
        Steps:
        1. Convert data to numpy aray
        2. Pick indices of k random point without replacement
        3. Find class (P) of each data point using euclidean distance_calc
        4. Stop when max_iteration are reached of P matrix doesn't change
        
        Return:
        np.array: containg class of each data point
        '''
        if isinstance(X, pd.DataFrame):X = X.values
        idx = np.random.choice(len(X), k, replace=False)
        centroids = X[idx, :]
        P = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
        for j in range(max_iterations):
            
            centroids = np.vstack([X[P==i,:].mean(axis=0) for i in range(k)])
            
            tmp = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
            
            if np.array_equal(P,tmp):
                break
            P = tmp
        P = P.tolist()

        nodes = initialize_data(self.dataframe)
        
        temp_list0 = []
        temp_list1 = []
        temp_list2 = []

        for i,node in enumerate(nodes):
            if P[i] == 0:
                temp_list0.append(node)
            elif P[i] == 1:
                temp_list1.append(node)
            elif P[i] == 2:
                temp_list2.append(node)                 
        self.model.append(temp_list0)
        self.model.append(temp_list1)
        self.model.append(temp_list2)
            
        print(self.model)
        #nodes = initialize_data(self.dataframe)
        
        # Get k random centers
        #centers = []
        #for i in range(self.k):
        #    centers.append(random.choice(nodes))
        #    
        #distance_calcs = []
        #for i,node in enumerate(nodes):
        #    distance_calcs.append(node, center, 
	#



        #for i in range(100):
        #                old_clustering = [0]
        #    new_clustering = [1]
	#
        #    first_iteration = 1
        #    while old_clustering != new_clustering:
        #        old_clustering = new_clustering
#
                # Choose k random centers
                #if first_iteration == 1:
                #    first_iteration = 0
                #    centers = []
                #    for integer in range(self.k):
                #        centers.append(random.choice(nodes))
                #
                #else:
                #    centers = []
                #    for cluster in old_clustering:
                #        centers.append(cluster.average())
            
                # Find the distance_calc between each node and the center
                #distance_calcs = []
                #for node1 in nodes:
                #    shortest_distance_calc = Shortest_distance_calc()
                #    for center in centers:
                #        if distance_calc(node1, center) < shortest_distance_calc.distance_calc and distance_calc(node1, center) != 0:
                #            if shortest_distance_calc.distance_calc != 0:
                #                shortest_distance_calc.update(distance_calc(node1, center), [node1], [center])
                #        distance_calcs.append(shortest_distance_calc)
                #
                # Create clusters
                #new_clustering = []
                #for center in centers:
                #    temp_cluster = []
                #    for distance_calc in distance_calcs:
                #        if center == distance_calc.cluster2:
                #            temp_cluster.append(distance_calc.cluster1)
                #    new_clustering.append(temp_cluster)


    # Average Linkage Algorithm
    def average_algorithm(self):

        # Initialize Singletons
        nodes = initialize_data(self.dataframe)
        for node in nodes:
            self.model.append(Cluster([node]))

        # Perform Algorithm
        while len(self.model) > self.k:
            shortest_distance_calc = Shortest_distance_calc()
            for i, cluster1 in enumerate(self.model[:-1]):
                for cluster2 in self.model[i+1:]:

                    if distance_calc(cluster1.average(), cluster2.average()) < shortest_distance_calc.distance_calc:
                        shortest_distance_calc.update(distance_calc(cluster1.average(), cluster2.average()), cluster1, cluster2) 
                
            # Merge Clusters
            self.model.remove(shortest_distance_calc.cluster1)
            self.model.remove(shortest_distance_calc.cluster2)
            self.model.append(Cluster(shortest_distance_calc.cluster1.nodes + shortest_distance_calc.cluster2.nodes))
        print(self.model)

    # Single Linkage Algorithm
    def single_algorithm(self):

        # Initialize Singletons
        nodes = initialize_data(self.dataframe)
        for node in nodes:
            self.model.append(Cluster([node]))

        # Perform Algorithm
        while len(self.model) > self.k:

            shortest_distance_calc = Shortest_distance_calc()
            for i, cluster1 in enumerate(self.model[:-1]):
                for node1 in cluster1.nodes:
                    for cluster2 in self.model[i+1:]:
                        for node2 in cluster2.nodes:
                            if distance_calc(node1, node2) < shortest_distance_calc.distance_calc:
                                shortest_distance_calc.update(distance_calc(node1, node2), cluster1, cluster2) 
                
            # Merge Clusters
            self.model.remove(shortest_distance_calc.cluster1)
            self.model.remove(shortest_distance_calc.cluster2)
            self.model.append(Cluster(shortest_distance_calc.cluster1 + shortest_distance_calc.cluster2))

class Shortest_distance_calc():
    def __init__(self):
        self.distance_calc = 9999999

    def update(self, distance_calc, cluster1, cluster2):
        self.distance_calc = distance_calc
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
        #string = "Key: " + str(self.key) + "\n"
        #for key in self.data:
        #    string = string + "\t" + str(key) + " -> " + str(self.data[key]) + "\n"
        #return string
        return str(self.key)

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

# Helper function to find the distance_calc between two nodes
def distance_calc(node1, node2):
    x = 0
    for key in node1.data:
        x = x + (node1.data[key] - node2.data[key])**2
        
    return math.sqrt(x)
    
# Calculate the differences between the clusters within the models and output the integer difference
def Hamming(model1, model2):
   hamming = [i for i in model1.model + model2.model if i not in model1.model or i not in model2.model]
   return len(hamming)/len(model1.clusters.nodes)
