
from sklearn.preprocessing import LabelEncoder

def labelEncoderColumns(df):
    df_kategorik = df.select_dtypes(include=["object"]).copy()
    df_columns = df_kategorik.columns
    for i in df_columns:
        lbe = LabelEncoder()
        df[i] = lbe.fit_transform(df[i])
    return df