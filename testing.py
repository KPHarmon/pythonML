from process import *

filename = "iris"

df = process_csv(filename)
model = Model(df, "AVERAGE", 5)
model.run()
