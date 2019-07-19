import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target

print(iris_X[:2,:])
print(iris_y)


X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y,
    test_size=0.3
)#将数据按0.7 和0.3分割，0.7 用于训练集，0.3用于测试集
knn=KNeighborsClassifier()#调用学习方法
knn.fit(X_train,y_train)#传入训练集
print(knn.predict(X_test))#传入测试集，输入测试集结果
print(y_test)#输入真实的测试集结果




























































