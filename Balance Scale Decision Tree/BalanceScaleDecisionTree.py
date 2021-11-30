import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def importData():
    '''Function to import data from csv

    Returns
    -------

    balance_data: Data from the csv file
    '''
    balance_data = pd.read_csv(r'balance-scale.data', sep= ',', header = None)
      
    # Printing the dataset shape
    print ("Dataset Length: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)
      
    # Printing the dataset obseravtions
    print ("Dataset: \n", balance_data.head())
    return balance_data

def dataSlicing(balance_data):
    '''Function to split the dataset into Training and Test Sets
    
    Params
    ------
    balance_data: DataFrame containing the entire dataset

    Returns
    --------
    X_train: List of attributes for Training set
    X_test: List of attributes for Test set
    Y_train: List of target values for Train set
    Y_test: List of target values for Test set
    
    '''
    X = balance_data.values[:, 1:5]        #Attributes from dataset
    Y = balance_data.values[:,0]           #Target Values for respective Atributes

    #Splitting the dataset into training and test sets
    #test_size parameter = 0.3 for 70:30 ratio b/w train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

    return X_train, X_test, y_train, y_test

def train_using_gini(X_train, X_test, y_train):
    '''Function to train the model using Gini Index

    Gini Index is a metric to measure 
    how often a randomly chosen element would be incorrectly identified.
    It means an attribute with lower gini index should be preferred.
    
    Params
    -------
    X_train: List of attributes for Training set
    X_test: List of attributes for Test set
    Y_train: List of target values for Train set

    Returns
    -------
    clf_gini: DecisionTreeClassifier object
    '''
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, X_test, y_train):
    '''Function to train the model using Entropy

    Entropy is the measure of uncertainty of a random variable, 
    it characterizes the impurity of an arbitrary collection of examples. 
    The higher the entropy the more the information content.
    
    Params
    -------
    X_train: List of attributes for Training set
    X_test: List of attributes for Test set
    Y_train: List of target values for Train set

    Returns
    -------
    clf_entropy: DecisionTreeClassifier object
    '''
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test, clf_object):
    '''Function to make predictions
    
    Params
    ------
    X_test: List of attributes for test set
    clf_object: DecisionTreeClassifier object
    
    Returns
    -------
    y_pred: List of Predicted values'''

    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    '''Function to calculate accuracy of prediction
    
    Params
    ------
    y_test: List of target values for Test set
    y_pred: List of Predicted values'''
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))

def main():
      
    # Building Phase
    data = importData()
    X_train, X_test, y_train, y_test = dataSlicing(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
      
    # Operational Phase
    print("Results Using Gini Index:")
      
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
      
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
      
      
# Calling main function
if __name__=="__main__":
    main()