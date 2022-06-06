# imported note:
# keras requirs one hot coded labels for multiclass problems
# skilearn can deal with integer classes for multiclass problems
# keras If you have two or more classes and  the labels are integers, the SparseCategoricalCrossentropy should be used. 
import os
import random
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)
# pip install --upgrade scikit-learn
# conda update -c conda-forge scikit-learn
# from sklearn.datasets import load_iris
import tensorflow as tf
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)

#make some random data
#With Tensorflow 2.0 this is reproducible code! 
reset_random_seeds()

import pandas as pd
# mspca_dwt_27x3600.csv
df = pd.read_csv (r'mspca_dwt_27x3600.csv', header=None)

df.head()
df.describe()
y = df[27]
y.head()
X = df.drop([27], axis=1)
X.head()



X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.2)
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)

#######################
### Neural Networks
#######################

# ('identity', 'logistic', 'tanh', 'relu')
#net = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_trainscaled, y_train) # %0.938 rs=4 .96
#net = MLPClassifier(hidden_layer_sizes=(64,32,16,32),activation="relu",random_state=1).fit(X_trainscaled, y_train) # %0.91
#net = MLPClassifier(hidden_layer_sizes=(128,64,32,32),activation="relu",random_state=1).fit(X_trainscaled, y_train) # %0.94 rs=4 .97
#net = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_trainscaled, y_train) # % 93.8 mspca_dwt_27x3600.csv .97
#net = MLPClassifier(hidden_layer_sizes=(256,128,64,32,32),activation="relu",random_state=1).fit(X_trainscaled, y_train) # % .94 rs=2 .95 rs=4 .97 
#net = MLPClassifier(hidden_layer_sizes=(128,256,128,32,16),activation="relu",
#                    random_state=1, verbose=10, solver='lbfgs',alpha=5).fit(X_trainscaled, y_train) # % .95 

#net = MLPClassifier(hidden_layer_sizes=(128,256,128,32,16),activation="relu",
#                    random_state=4).fit(X_trainscaled, y_train) # % .96 rs=2 .97 rs=4 .98 
#net = MLPClassifier(hidden_layer_sizes=(128,256,128,64,16),activation="relu",random_state=1).fit(X_trainscaled, y_train) # % .96 rs=2 .97 rs=4 .98
#net = MLPClassifier(hidden_layer_sizes=(32,128,128,64,32),activation="relu",random_state=1).fit(X_trainscaled, y_train) # % .93 rs=2 .97 rs=4 .97
#net = MLPClassifier(hidden_layer_sizes=(32,64,64,32,8),activation="relu",random_state=1).fit(X_trainscaled, y_train) # % .91 rs=2 .97 rs=4 .98 
#net = MLPClassifier(hidden_layer_sizes=(32,64,32,16,8),activation="relu",random_state=1).fit(X_trainscaled, y_train) # % .  rs=2 .  rs=4 .97
#net = MLPClassifier(hidden_layer_sizes=(32,64,64,64,32),activation="relu",random_state=4).fit(X_trainscaled, y_train) # % .  rs=2 .  rs=4 .98  mspca_dwt_27x3600.csv rs=4 .98 rs=2 .98 rs=1 .98
#net = MLPClassifier(hidden_layer_sizes=(32,64,64,64,32),activation="relu",max_iter=500, alpha=0.0001,
#                     solver='sgd', verbose=10,  random_state=1,tol=0.000000001).fit(X_trainscaled, y_train) # % .  rs=2 .  rs=4 .97  

#net = MLPClassifier(hidden_layer_sizes=(128,256,128,16),activation="relu",
#                    random_state=4).fit(X_trainscaled, y_train) # %  rs=4 .986 

#net = MLPClassifier(hidden_layer_sizes=(64,256,128,16),activation="relu",
#                    random_state=4).fit(X_trainscaled, y_train) # % .96 rs=2  rs=4 .987

#net = MLPClassifier(hidden_layer_sizes=(128,256,100,16),activation="relu",
#                    random_state=2).fit(X_trainscaled, y_train) # % .96 rs=2  rs=2 .987

#net = MLPClassifier(hidden_layer_sizes=(64,256,128,16),activation="relu",
#                    random_state=4).fit(X_trainscaled, y_train) # % .96 rs=2  rs=4 .9875

#net = MLPClassifier(hidden_layer_sizes=(32,64,128,64,16),activation="relu",
#                    random_state=1).fit(X_trainscaled, y_train) # % .96 rs=2  rs=4 .983

#net = MLPClassifier(hidden_layer_sizes=(64,32,64,16,32),activation="relu",
#                    random_state=1).fit(X_trainscaled, y_train) # % .96 rs=2  rs=4 .983

#net = MLPClassifier(hidden_layer_sizes=(64,32,256,3,3),activation="relu",
#                    random_state=1).fit(X_trainscaled, y_train) # % .96 rs=2  rs=4 .0.986

#net = MLPClassifier(hidden_layer_sizes=(512,16,32,3,3),activation="relu",
#                    random_state=1).fit(X_trainscaled, y_train) # % .96 rs=2  rs=4 .0.988888
#net = MLPClassifier(hidden_layer_sizes=(60,30,15,5,3),activation="relu",
#                    random_state=1,max_iter=500).fit(X_trainscaled, y_train) # % .96 rs=2  rs=4 .0.986
net = MLPClassifier(hidden_layer_sizes=(256,64,15,3),activation="logistic", # logistic, tanh
                    random_state=1,max_iter=500).fit(X_trainscaled, y_train) # % .96 rs=2  rs=4 .0.
net
#net.out_activation_ = 'logistic'
#net.activation='tanh'
y_pred=net.predict(X_testscaled)
print(net.score(X_testscaled, y_test))
#fig=confusion_matrix(net, X_testscaled, y_test,display_labels=["Setosa","Versicolor","Virginica"])
#fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
#plt.show()


cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
print(net.out_activation_)

#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, net.predict(X_testscaled)))
#print(accuracy_score(y_test, y_pred))

plt.plot(net.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import model_selection


tuned_parameters = {
    'hidden_layer_sizes': [(10,),(30,),(60,),(100,),(110,),(120,),(130,),(150,),(10,20),(10,30),(10,50),(30,30),(90,30),(90,10),(10,5,3),(10,3,3),(30,20,10),(30,20,3),(256,64,30),(150,50,30),(150,50,10),(150,10,30),(150,100,30),(256,64,15,3),(256,128,64,32),(256,128,64,32,32),(128,256,128,64,16),(32,64,64,64,32),(32,64,64,64,32),(512,16,32,3,3),(32,64,128,64,16),(60,30,15,5,3)],
    'activation': ['logistic','tanh'], # 'relu'
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [0.0001, 0.1, 0.5],
    'learning_rate': ['constant','adaptive'],
}




from sklearn.metrics import accuracy_score



clf = GridSearchCV(MLPClassifier(random_state=1), param_grid=tuned_parameters, n_jobs=-1, cv=3, verbose=5)
clf.fit(X_trainscaled,y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

    

print('Train Accuracy : %.3f'%clf.best_estimator_.score(X_trainscaled, y_train))
print('Test Accuracy : %.3f'%clf.best_estimator_.score(X_testscaled, y_test))
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)

## https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-neural-network
#from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(Y_test, Y_preds):
    conf_mat = confusion_matrix(Y_test, Y_preds)
    #print(conf_mat)
    fig = plt.figure(figsize=(6,6))
    plt.matshow(conf_mat, cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(3), range(3))
    plt.xticks(range(3), range(3))
    plt.colorbar();
    for i in range(3):
        for j in range(3):
            plt.text(i-0.2,j+0.1, str(conf_mat[j, i]), color='tab:red')

plot_confusion_matrix(y_test, clf.best_estimator_.predict(X_testscaled))
plt.show()
[weights.shape for weights in clf.best_estimator_.coefs_]
print("Name of Output Layer Activation Function : ", clf.best_estimator_.out_activation_)
print("Number of Iterations for Which Estimator Ran : ", clf.best_estimator_.n_iter_)
print()

