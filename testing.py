from process import *

filename = "iris"

df = process_csv(filename)
model = Model(df, "LLOYD", 5)
model.run()
