
import pandas as pd
import numpy as np

features = [ "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
df = pd.read_csv("C:/Users/daksh/code/credit approval model/credit+approval/crx.data",header=None, names = features)
df.replace("?", np.nan, inplace=True)
cat_cols = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]

for col in cat_cols:
    print(f"{col}: {df[col].dropna().unique().tolist()}")
