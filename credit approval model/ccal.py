import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


features = [ "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
df = pd.read_csv("C:/Users/daksh/code/credit approval model/credit+approval/crx.data",header=None, names = features)
df.replace("?", np.nan, inplace=True)
print(f"Missing values found = {df.isna().values.any()}")
X = df.drop("A16", axis=1)
y = df["A16"]
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"),cat_cols )
    ]
) 

model = Pipeline(steps=[
    ("preprocessor", preprocess),
    ("classifier", LogisticRegression( class_weight="balanced"))
])

param_grid = {
    "classifier__C": [0.01, 0.1, 0.5, 1, 2, 5, 10]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=5
)

grid.fit(X_train,y_train)

para = grid.best_params_["classifier__C"]
print(f"bestparameterforcis = {para}")