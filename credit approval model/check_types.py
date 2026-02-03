
import pandas as pd
import numpy as np

features = [ "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
df = pd.read_csv("C:/Users/daksh/code/credit approval model/credit+approval/crx.data",header=None, names = features)
df.replace("?", np.nan, inplace=True)
X = df.drop("A16", axis=1)

print("Dtypes before processing:")
print(X.dtypes)

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns
print("\nNum cols detected:", num_cols)
print("Cat cols detected:", cat_cols)
