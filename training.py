import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


dataset= pd.read_csv("final_dataset.csv")

#here i am seperating the features and the label , which is the yoga pose name
X= dataset.drop("Label", axis=1)
y= dataset["Label"]

X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2 , random_state=100)

model= KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

prediction_pose=model.predict(X_test)

accuracy=accuracy_score(y_test,prediction_pose)
print(accuracy)

test_data = [[135.0,50,80, 145.0, 90.0, 85.0, 180.0, 175.0, 160.0, 158.0, 100.0, 90.0, 95.0, 80.0]]  # shape (1, n_features)
prediction=model.predict(test_data)
print(prediction)


def conf_matrix(y_test, prediction_pose):
    matrix = confusion_matrix(y_test, prediction_pose)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=plt.cm.Reds)
    plt.title("Confusion Matrix")
    plt.show()
