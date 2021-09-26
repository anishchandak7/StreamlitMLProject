# Imports
import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Set the Web Apps title
st.title("Streamlit Machine Learning Project")

# Give small description using HTML Markup
st.write("""
### To Examine and Explore Different Classifiers on various datasets

Which classifier is the best?
""")

# Selectbox to select dataset
dataset_name = st.sidebar.selectbox(label="Select a dataset",options=("Iris","Wine","Breast cancer"))

# Selectbox to select classifier
classifier_name = st.sidebar.selectbox(label = "Select a classifier",options = ("KNN","SVM","Random Forest")) 

# Load the dataset
def get_dataset(dataset_name):
    if dataset_name.lower() == "iris":
        data = datasets.load_iris()
    elif dataset_name.lower() == "wine":
        data = datasets.load_wine()
    elif dataset_name.lower() == "breast cancer":
        data = datasets.load_breast_cancer()
    
    X = data.data
    y = data.target
    return X,y

X,y = get_dataset(dataset_name)
st.write("Shape of dataset:",X.shape)
st.write("Number of classes:",len(np.unique(y)))

def add_parameter_ui(classifier_name):
    
    params = {}
    if classifier_name.lower() == "knn":
        K = st.sidebar.slider(label = "K",min_value=1,max_value=15)
        params["K"] = K
    elif classifier_name.lower() == "svm":
        C = st.sidebar.slider(label = "C",min_value = 0.01,max_value = 10.0)
        params["C"] = C
    elif classifier_name.lower() == "random forest":
        max_depth = st.sidebar.slider(label = "Max depth",min_value = 2, max_value = 15)
        n_estimators = st.sidebar.slider(label = "Estimators",min_value = 1,max_value = 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

def get_classifier(classifier_name,params):

    if classifier_name.lower() == "knn":
        classifier = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name.lower() == "svm":
        classifier = SVC(C=params["C"])
    elif classifier_name.lower() == "random forest":
        classifier = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return classifier


params = add_parameter_ui(classifier_name)
classifier = get_classifier(classifier_name,params)

# Classification
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 1234)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write(f"classifier = {classifier_name}")
st.write(f"Accuracy = {accuracy}")

# PCA Plotting
pca = PCA(2) # 2 - Dim

X_estimators = pca.fit_transform(X)
x1 = X_estimators[:,0]
x2 = X_estimators[:,1]

fig = plt.figure()
plt.scatter(x1, x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)

