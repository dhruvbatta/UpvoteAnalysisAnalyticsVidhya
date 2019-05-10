import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')


df=df.drop('ID',axis=1)
print(df.head())
#print (df['Tag'])

ytrain=df.pop('Upvotes')
print(ytrain.head())
xtrain=df
print(xtrain.head())

print(xtrain.Tag.value_counts())
xtrain.Tag = xtrain.Tag.astype('category')
#print(xtrain.head(10))

dftest.Tag = dftest.Tag.astype('category')

class ModifiedLabelEncoder(LabelEncoder):

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)
# Ref.: https://stackoverflow.com/questions/48994618/unable-to-use-featureunion-to-combine-processed-numeric-and-categorical-features
from sklearn.base import BaseEstimator, TransformerMixin

# Class that identifies Column type
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit (self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

numeric_cols = ['Reputation', 'Answers', 'Username', 'Views'] # list of numeric column names
categorical_cols = ['Tag'] # list of categorical column names

# Testing
print(ColumnSelector(columns=numeric_cols).fit_transform(xtrain).head())
print(ColumnSelector(columns=categorical_cols).fit_transform(xtrain).head())



numeric_cols_pipe = make_pipeline(ColumnSelector(columns=numeric_cols),StandardScaler())
categorical_cols_pipe = make_pipeline(ColumnSelector(columns=categorical_cols), ModifiedLabelEncoder(), OneHotEncoder(sparse=False))

fu = make_union(numeric_cols_pipe, categorical_cols_pipe)

trans_vec = fu.fit_transform(xtrain)
print(trans_vec.shape)
print(trans_vec[:5])

#Using Principal Component analysis
pca = PCA(n_components=5)
principalComponents = pca.fit(trans_vec)

pca_vec = pca.transform(trans_vec)
test_fu_vec = fu.transform(dftest)
print(test_fu_vec.shape)


test_pca_vec = pca.transform(test_fu_vec)
print(test_pca_vec.shape)

x_train, x_test, y_train, y_test = train_test_split(pca_vec, ytrain.values, train_size = 0.66, random_state = 0)




