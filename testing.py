from process import *

filename = "iris"

df = process_csv(filename)
model = Model(df, "LLoyd", 3)
model.run()
# print(Hamming(model.model))
