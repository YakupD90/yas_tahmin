import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import ml_xgboost

#data seti oku
"""bu data setde parametre depğerinin tamamının  aynı olduğu sutunlar ve
 yaş aynı olmasına rağmen parametreler 
 arasında fazla fark olan sutunlar kaldırılmıştır"""
df=pd.read_excel("sonKopya.xlsx")

bagımsız=df.drop(["Age"],axis=1)
bagımlı=df["Age"]

bagımsız_train,bagımsız_test,bagımlı_train,bagımlı_test=train_test_split(bagımsız,bagımlı,test_size=0.2, random_state=1)
model=RandomForestRegressor()
model.fit(bagımsız_train,bagımlı_train)
tahmin=model.predict(bagımsız_test)

print("Random Forest")

mae = mean_absolute_error(bagımlı_test, tahmin)
print(f"Ortalama mutlak deger(MAE): {mae}")

# R-kare hesaplama
r2 = r2_score(bagımlı_test,tahmin)
print(f"R-kare skoru: {r2}\n")

