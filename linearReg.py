import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot

from batchlearning import y_pred


def cost_compute(x,y,theta):
    m=len(y)
    pred=x.dot(theta)
    val=pred-y
    cost=(1/(2*m))* np.sum((val**2))
    return cost



data={'temp':[10,15,20,25,30,35,40],'sales':[100,150,200,250,300,350,400]}
dataset=pd.DataFrame(data)
x=dataset[['temp']]
y=dataset[['sales']]
# print(dataset)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("the total sles is:",y_pred[0][0])
X = np.array([[1, 20], [1, 25], [1, 30]])  # Add 1 for bias term
y = np.array([40, 50, 60])

theta = np.array([0, 2])   # Example parameters (θ0=0, θ1=2)

# Compute cost
cost =cost_compute(X,y, theta)
print("Cost:", cost)
from sklearn.metrics import r2_score

print("R² Score:", r2_score(y_test, y_pred))

plt.scatter(x_train, y_train, color="blue")
plt.plot(x_train, model.predict(x_train), color="red")
plt.xlabel("Temperature")
plt.ylabel("Sales")
plt.show()


