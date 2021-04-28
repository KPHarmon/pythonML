from process import *

filename = "iris"

df = process_csv(filename)
nodes = initialize_data(df)
print(nodes[0])
print(nodes[149])
