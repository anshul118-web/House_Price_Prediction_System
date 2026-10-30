import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("dataset/house_data.csv")

print("\n===== DATASET INFORMATION =====")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
df.info()


print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


plt.figure(figsize=(8, 5))

sns.histplot(df["Price"], kde=True)

plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")

plt.show()


plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")

plt.show()

print("\nData Analysis Completed!")
