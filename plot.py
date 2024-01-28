import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output.csv")

plt.figure(figsize=(10, 6))
columns = ["matmul in_proj", "conv", "ssm", "matmul out_proj", "matmul logits"]

df = pd.DataFrame(df.iloc[:, :].values, columns=columns)

print(df.describe())

for i, col in enumerate(columns):
    plt.plot(df.index, df.iloc[:, i], label=col)

plt.title("Time Taken for a Token Position")
plt.xlabel("Token Position")
plt.ylabel("Time (in seconds)")
plt.legend()
plt.grid(True)
plt.show()
