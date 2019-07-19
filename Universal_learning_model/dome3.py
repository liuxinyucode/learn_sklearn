from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()

model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))

#y=5*x+9
print(model.coef_)#5
print(model.intercept_)#9
print(model.get_params())
print(model.score(data_X,data_y))#打分: R^2 coefficient of determination















































