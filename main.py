import pandas as pd;
import numpy as np;
import matplotlib.pylab as plt

df = pd.read_csv("A24/A24.csv")

columns = list(df.columns.values)[16:]

df = df[columns] 

columns_lum = columns[:-1]
columns_lum.remove("Luminosity_Index")
df_lum = df[columns_lum].dropna()

df_lum_nan = df[df["Luminosity_Index"].isnull()]
df_lum_nan = df_lum_nan.drop("Luminosity_Index", axis=1)
df_lum_nan = df_lum_nan.drop("class", axis=1)
df_sum = df_lum.apply(np.sum, axis=1, result_type="reduce")


def artimiausia_lum(val, i):
    new_df = df_sum.apply(lambda x: abs(x - val))
    sorted = new_df.sort_values(ascending=True)
    df.at[i, "Luminosity_Index"] = df.at[sorted.index[1], "Luminosity_Index"]

for index, row in df_lum_nan.iterrows():
    artimiausia_lum(row.sum(), index)


columns_sig = columns[:-1]
columns_sig.remove("SigmoidOfAreas")
df_sig = df[columns_sig].dropna()

df_sig_nan = df[df["SigmoidOfAreas"].isnull()]
df_sig_nan = df_sig_nan.drop("SigmoidOfAreas", axis=1)
df_sig_nan = df_sig_nan.drop("class", axis=1)
df_sum = df_sig.apply(np.sum, axis=1, result_type="reduce")

def artimiausia_sig(val, i):
    new_df = df_sum.apply(lambda x: abs(x - val))
    sorted = new_df.sort_values(ascending=True)
    df.at[i, "SigmoidOfAreas"] = df.at[sorted.index[1], "SigmoidOfAreas"]

for index, row in df_sig_nan.iterrows():
    artimiausia_sig(row.sum(), index)

# df[:-1].hist() # histogramos
# plt.show()

# df.boxplot() # isskirtys pries
# plt.show()

def robust_scaling(column):
    mediana = df[column].median()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df[column] = df[column].apply(lambda x: (x - mediana) / IQR)

robust_scaling("LogOfAreas")
robust_scaling("Log_X_Index")
robust_scaling("Log_Y_Index")
robust_scaling("Luminosity_Index")

# df_dalis.boxplot() # isskirtys po
# plt.show()
