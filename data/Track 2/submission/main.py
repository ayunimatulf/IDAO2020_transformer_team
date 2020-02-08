import pandas as pd

toc=time.time()
test_mydata = pd.read_csv("test_mydata.csv")
print(test_mydata.head())
train_stats=pd.read_csv('train_stats.csv')
print(train_stats.head())
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
label=["id", "x", "y", "z", "Vx", "Vy", "Vz"]
normed_test_data = norm(test_mydata.drop(label, axis=1))
tic=time.time()
print(tic-toc)
# print(test_mydata.head)
print(normed_test_data.head())
# just sending simulated values as the answer
# submission = test[["id", "x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]]
# submission.columns = ["id", "x", "y", "z", "Vx", "Vy", "Vz"]
# submission.to_csv("submission.csv", index=False)