import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
"""bu data setde parametre depğerinin tamamının  aynı olduğu sutunlar ve
 yaş aynı olmasına rağmen parametreler 
 arasında fazla fark olan sutunlar kaldırılmıştır"""

#data seti oku
df=pd.read_excel("sonKopya.xlsx")

#değerlerinin tamamı aynı olan sutnları sil, ve kaydet
"""
df=pd.read_excel("sonKopya.xlsx")
print(df.head())
sil= df.loc[:, (df != df.iloc[0]).any()]#aynı olanlar silindi
sil.to_excel("sonKopya.xlsx")
df2=pd.read_excel("sonKopya.xlsx")
print(df2.head())"""

bagımsız=df.drop(["Age"],axis=1)
bagımlı=df["Age"]

bagımsız_train,bagımsız_test,bagımlı_train,bagımlı_test=train_test_split(bagımsız,bagımlı,test_size=0.2, random_state=10)
model=xgb.XGBRegressor(colsample_bytree=0.3, learning_rate=0.1, max_depth=3, n_estimators=100)
model.fit(bagımsız_train,bagımlı_train)
tahmin=model.predict(bagımsız_test)
print("XGBoost")
mae = mean_absolute_error(bagımlı_test, tahmin)
print(f"Ortamala mutlak hata (MAE): {mae}")

# R-kare hesaplama
r2 = r2_score(bagımlı_test,tahmin)
print(f"R-kare: {r2}\n")