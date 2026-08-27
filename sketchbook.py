import pandas as pd


train = pd.read_csv("data/train.csv")

train["il"] = train["lokasyon"].str.split(">").str[0]
train["bölge"] = train["lokasyon"].apply(lambda x: x.split(">")[1] if len(x.split(">")) == 3 else None)
train["ilçe"] = train["lokasyon"].apply(lambda x: x.split(">")[1] if len(x.split(">")) == 2 else (x.split(">")[2]))

print(train[train["il"] == "MANİSA"]["bölge"].nunique())