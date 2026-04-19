import pandas as pd

df = pd.read_parquet("datos/SPO-PT03083_06001_100.parquet")

print("--------------------------------")
print(df["Samplingpoint"].unique())
df["Start"] = pd.to_datetime(df["Start"])
print(df["Start"].min(), df["Start"].max())
df = df[df["Validity"] == 1]
df["date"] = df["Start"].dt.date

daily = df.groupby("date")["Value"].mean().reset_index()

print(daily.head())
print(len(daily))


#print(df.columns)
#print(df.head())
#print(df.shape)

