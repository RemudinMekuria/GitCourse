# Test accuracy: 0.9888888597488403

import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.constraints import max_norm
import tensorflow.keras.layers 
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)

#make some random data
#With Tensorflow 2.0 this is reproducible code! 
reset_random_seeds()

# load and prepare the dataset

df = pd.read_csv (r'mspca_dwt_27x3600.csv', header=None)
df.head()
df.describe()
y = df[27]
# convert integers to dummy variables (i.e. one hot encoded)
from tensorflow.keras.utils import to_categorical

# from keras.utils import np_utils
# binary encode
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y = y.values
y = y.reshape(len(y), 1)
y = onehot_encoder.fit_transform(y)
print(y)

X = df.drop([27], axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.2)
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)


input = X_trainscaled
target= y_train


def myfit(x, t):
    reset_random_seeds()
    # 60,30,15,5,3
    model = keras.Sequential([
            keras.layers.Dense(120, input_dim=x.shape[1], 
                               #activation='relu',
                               kernel_regularizer=keras.regularizers.l2(l=0.000001),# 0.0005 dene
                               bias_regularizer=keras.regularizers.l2(l=0.000001)), #60
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(), #LeakyReLU, ReLU 
            #keras.layers.Dropout(0.3), #yok
            keras.layers.Dense(75, 
                               #activation='relu',
                               bias_regularizer=keras.regularizers.l2(l=0.000001),
                               kernel_regularizer=keras.regularizers.l2(l=0.000001),
                               kernel_constraint=max_norm(3) #unit_norm()
                               ), # 30
            #keras.layers.BatchNormalization(),
            keras.layers.ReLU(), #LeakyReLU, ReLU 
            keras.layers.Dropout(0.4), #yok
            keras.layers.Dense(30, 
                               #activation='relu',
                               bias_regularizer=keras.regularizers.l2(l=0.000001),
                               kernel_regularizer=keras.regularizers.l2(l=0.000001)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(), #LeakyReLU, ReLU 
            keras.layers.Dense(5, activation='relu'),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(3, activation='softmax')
        ])
    
    opt = keras.optimizers.Adam(
        #learning_rate=0.01
        )
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

    # or pass optimizer by name: default parameters will be used
    # model.compile(loss='categorical_crossentropy', optimizer='adam')

    validation_split = 0.1 #.1
    batch_size=200 # 200 30 also good 100 98.75
    no_epochs=150 # 300 e:150 b:200 good
    verbosity = 2
    reset_random_seeds()
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=5)

    callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-3 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 10 epochs"
        patience=5,
        verbose=1)
    ]

    
    history = model.fit(x, t,
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity,
              shuffle=True,
              validation_split=validation_split,
              #validation_data=(X_test, y_test),
              #callbacks=[callbacks]
              )
    #model.fit(x, t, epochs=NUM_EPOCHS, verbose=1)
    y_pred = model.predict(X_testscaled)
    #Your input to confusion_matrix must be an array of int not one hot encodings.
    cf=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cf)
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cf,
                     index = ['Normal','Myopathy','ALS'], 
                     columns = ['Normal','Myopathy','ALS'])

    #Plotting the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True,fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()


    #loss = model.evaluate(x,  t) #This prints out the loss by side-effect
    score = model.evaluate(X_testscaled, y_test, verbose=1)
    loss, accuracy = model.evaluate(X_testscaled, y_test)
    print(f'Test loss for Keras ReLU : {score[0]} / Test accuracy: {score[1]}')
    return history, model


history, model=myfit(input, target)


## Visualize model history

dataset = pd.DataFrame({'accuracy':history.history['accuracy'],
                        'val_accuracy':history.history['val_accuracy'],
                        'loss':history.history['loss'],
                        'val_loss':history.history['val_loss'],
                        'Epoch':list(range(1, len(history.history['accuracy'])+1)),
                        
                   })
 
# creating axes object and defining plot
ax = dataset.plot(kind = 'line', x = 'Epoch',
                  y = ['accuracy','val_accuracy'], color = ['blue','red'],
                  linewidth = 1)

ax2 = dataset.plot(kind = 'line', x = 'Epoch',
                   y = 'loss', secondary_y = True,
                   color = 'blue',  linewidth = 1,
                   ax = ax)
ax2 = dataset.plot(kind = 'line', x = 'Epoch',
                   y = 'val_loss', secondary_y = True,
                   color = 'red',  linewidth = 1,
                   ax = ax)
#title of the plot
plt.title("Learning curves")
#plt.legend(loc='center right')
 
#labeling x and y-axis
ax.set_xlabel('Epoch', color = 'g')
ax.set_ylabel('accuracy', color = "b")
ax2.set_ylabel('loss', color = 'r')
 
#defining display layout
plt.tight_layout()
# Adding legend
ax.legend(loc='center right')
ax2.legend(loc='center',bbox_to_anchor=(0.5, 0.5))

#show plot
plt.show()


# importing the module
import pandas as pd
y_pred = model.predict(X_testscaled)
# creating the DataFrame
df_data = pd.DataFrame({'targets': y_test.argmax(axis=1),
                           'predictions': y_pred.argmax(axis=1),
                          })
  
# determining the name of the file
file_name = 'resultsData.xlsx'
  
# saving the excel
df_data.to_excel(file_name)


print(model.summary())
