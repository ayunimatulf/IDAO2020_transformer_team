import pandas as pd

test = pd.read_csv("test.csv")

# just sending simulated values as the answer
submission = test[["id", "x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]]
submission.columns = ["id", "x", "y", "z", "Vx", "Vy", "Vz"]
submission.to_csv("submission.csv", index=False)