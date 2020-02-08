# Startegi : Using each label using neural network(regression) and choose predictor by correlation (change correlation look as a absolute number with auto threshold 0.02 
## epoch -> 100

import seaborn as sb
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)


path='../input/train.csv'
# path='data/train.csv'
list_variable={'x':['x_sim', 'z_sim', 'Vy_sim'],
         'y': ['x_sim', 'y_sim', 'Vz_sim'],
         'z': ['sat_id', 'x_sim', 'z_sim', 'Vy_sim', 'Vz_sim']}#,
        #  'Vx': ['y_sim', 'Vx_sim', 'Vz_sim'],
        #  'Vy': ['z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim'],
        #  'Vz': ['y_sim', 'z_sim', 'Vy_sim', 'Vz_sim']}
#read daraframe and copy
df=pd.read_csv(path, index_col=0)
df=df.drop('epoch', axis=1)
mydata=df.copy()

#split train test
train_mydata = mydata.sample(frac=0.8,random_state=0)
test_mydata = mydata.drop(train_mydata.index)
test_mydata.head()
print('train_mydata.shape : %s, test_mydata.shape : %s'%(train_mydata.shape, test_mydata.shape))

#get statistical info
label=['Vx','Vy','Vz','x','y','z']
train_stats = train_mydata.describe()
for i in label:
  train_stats.pop(i)
train_stats = train_stats.transpose()
train_stats
#save stats info
train_stats.to_csv('train_stats.csv')

#split label
train_labels = train_mydata[label]
test_labels = test_mydata[label]
print(train_labels.head(), test_labels.head())

#normalisize
def norm(x):
  try:
    return (x - train_stats['mean']) / train_stats['std']
  except:
    print(x)
normed_train_data = norm(train_mydata.drop(label, axis=1))
normed_test_data = norm(test_mydata.drop(label, axis=1))
print('normed_test_data.shape', normed_test_data.shape)

# model function 
def create_model(input_dim):
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(100, kernel_initializer='normal',input_dim = input_dim, activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(200, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(200, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(200, kernel_initializer='normal',activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()
    return NN_model

def create_checkpoint(variable):
  filename='8_model_'+variable+'.h5'
  checkpoint_callback = ModelCheckpoint(filename
    , monitor='val_mean_absolute_error', verbose=1,
    save_best_only=True, save_weights_only=False, mode='min')
  return checkpoint_callback 

for i in list_variable:
    train=normed_train_data[list_variable[i]]
    label_train=train_labels[i].values
    print(i, train.columns)
    test=normed_test_data[list_variable[i]]
    label_test=test_labels[i].values
    print(len(train.columns))
    model=create_model(input_dim=len(train.columns))
    checkpoint_callback=create_checkpoint(i)
    model.fit(train, label_train, validation_data=(test, label_test), epochs=100, callbacks=[checkpoint_callback])
    fn=path+i+'_model_100_8.h5'
    model.save(fn)
    print('succes save to : %s.'%fn)
    test_mydata[i+'_pred']=model.predict(test)

def smape(satellite_predicted_values, satellite_true_values):
    return np.mean(np.abs((satellite_predicted_values - satellite_true_values)/(np.abs(satellite_predicted_values) + np.abs(satellite_true_values))))

for i in ['_sim', '_pred']:
    real=test_mydata[label].values
    comp=test_mydata[[j+i for j in label]].values
    print('smape real vs %s : %f'%(i, smape(real, comp)))
