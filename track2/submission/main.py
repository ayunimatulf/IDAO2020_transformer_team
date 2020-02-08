import pandas as pd
from tensorflow.keras import models
import time


#data stats for normalization
df_stats=pd.read_csv('train_stats.csv', index_col=0)
df_stats=df_stats.drop('sat_id')

#data variable
list_variable={'x':['x_sim', 'z_sim', 'Vy_sim'],
         'y': ['x_sim', 'y_sim', 'Vz_sim'],
         'z': ['x_sim', 'z_sim', 'Vy_sim', 'Vz_sim'],
         'Vx': ['y_sim', 'Vx_sim', 'Vz_sim'],
         'Vy': ['z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim'],
         'Vz': ['y_sim', 'z_sim', 'Vy_sim', 'Vz_sim']}

#normalization
def norm(x):
    return (x - df_stats['mean']) / df_stats['std']
test = pd.read_csv("/Users/bytedance-it138302/Documents/IDAO2020/test_2.csv")
# just sending simulated values as the answer
submission = test[["id", "x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]]
submission=submission.set_index('id')
normed_data=norm(submission)
for i in list_variable:
    model=models.load_model('model/8_model_'+i+'.h5')
    submission[i]=model.predict(normed_data[list_variable[i]].values)
submission = submission[["x", "y", "z", "Vx", "Vy", "Vz"]]
submission.to_csv("submission.csv", index=True)