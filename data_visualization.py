
import seaborn as sns
import matplotlib.pyplot as plt

def visualization(columns, df):
    for i in columns:
        plt.figure(figsize=(15, 6))
        sns.countplot(x=df[i], data=df)
        plt.show()