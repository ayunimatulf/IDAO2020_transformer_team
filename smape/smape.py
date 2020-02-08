import pandas as pd 
import numpy as np 

def smape(satellite_predicted_values, satellite_true_values):
    return np.mean(np.abs((satellite_predicted_values - satellite_true_values)/(np.abs(satellite_predicted_values) + np.abs(satellite_true_values))))

pred=pd.read_csv('/Users/bytedance-it138302/Documents/IDAO2020/track2/submission.csv')
real=pd.read_csv('/Users/bytedance-it138302/Documents/IDAO2020/test_label_real.csv')
sim=pd.read_csv('/Users/bytedance-it138302/Documents/IDAO2020/test_2.csv')
print(sim.head())

label=['x','y','z','Vx', 'Vy', 'Vz']
re=real[label].values
prediction=pred[label].values
print('smape real vs prediction : %f'%smape(prediction, re))

re=real[label].values
prediction=sim[[i+'_sim' for i in label]].values
print('smape real vs simulation: %f'%smape(prediction, re))