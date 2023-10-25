#%% load boston database
from sklearn.datasets import load_boston
X ,y = load_boston(return_X_y=True)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
mod = KNeighborsRegressor()
mod.fit(X, y)
ModPredict1=mod.predict(X)
len1= len(ModPredict1)
len2= len(X)
ModPredict1
# print (len1)
# print (len2)
# %%
mod=LinearRegression()
mod.fit(X, y)
ModPredict1=mod.predict(X)
ModPredict1
# %%
X ,y = load_boston(return_X_y=True)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt
from icecream import ic
mod = KNeighborsRegressor().fit(X , y)
pipe= Pipeline([
    ("scale",StandardScaler()),
    ("model",KNeighborsRegressor(n_neighbors=1))
])
pipe.fit(X ,y)
pred=mod.predict(X) 
plt.figure()
plt.scatter(pred, y)
pred=pipe.predict(X) 
plt.figure()
plt.scatter(pred, y, c="red")

