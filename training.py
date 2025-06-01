import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from sklearn.model_selection import cross_val_score
import seaborn as sns

dataset= pd.read_csv("dataset_images.csv")

#here i am seperating the features and the label , which is the yoga pose name
X= dataset.drop("Label", axis=1)
y= dataset["Label"]

X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2 )

model= KNeighborsClassifier(n_neighbors=3)

model.fit(X_train,y_train)

prediction_pose=model.predict(X_test)

accuracy=accuracy_score(y_test,prediction_pose)
print(accuracy)




print(cross_val_score(model, X,y))
joblib.dump(model, "mymodel.pkl")


def conf_matrix(y_test, prediction_pose):
    matrix = confusion_matrix(y_test, prediction_pose)
    sns.heatmap(matrix, square = True,annot = True,cbar = False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Pose")
    plt.ylabel("True Pose")
    plt.show()

conf_matrix(y_test,prediction_pose)
