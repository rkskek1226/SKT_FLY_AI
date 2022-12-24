import my_utils as my
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   # 정확도
from sklearn.metrics import precision_score   # 정밀도
from sklearn.metrics import recall_score   # 재현율
from sklearn.metrics import confusion_matrix


def get_iris(mode=None):
    iris = pd.read_csv("../data/iris.csv")
    df = iris.drop(["Id"], axis=1).copy()
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    if mode == "bin":
        df = df.loc[df["species"] != "Iris-virginica"]
    
    df["species"] = df["species"].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})   # Label Encoding
    
    # 데이터 분리
    x_data = df.drop(["species"], axis=1)
    y_data = df.iloc[:, -1] 
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
    
    return x_train, x_test, y_train, y_test


def print_score(y_test, y_pred, average="binary"):
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average=average)   # 다중분류때는 average에 "macro"
    rec = recall_score(y_test, y_pred, average=average)   # 다중분류때는 average에 "macro"
    print("accuracy :", acc)
    print("precision :", pre)
    print("recall :", rec)

    
def plpt_confusion_matrix(y_test, y_pred):
    cfm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cfm, annot=True, cbar=False)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()