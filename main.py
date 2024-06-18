import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Membaca data dari file sales.txt dengan menambahkan header secara manual
data = pd.read_csv('data/sales.txt', delim_whitespace=True, header=None, names=['Sales', 'Advertising'])

# Memeriksa apakah data telah dibaca dengan benar
print("Data yang telah dibaca:")
print(data.head())

# Memeriksa kolom yang terbaca
print("Kolom dalam data:")
print(data.columns)

# Memisahkan fitur dan target
X = data[['Sales']]
y = data['Advertising']

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Melakukan prediksi
y_pred = model.predict(X_test)

# Menampilkan koefisien regresi
print(f'Koefisien regresi: {model.coef_[0]}, Intercept: {model.intercept_}')

# Memprediksi cost advertising untuk 50, 100, dan 150 sales
sales_values = [[50], [100], [150]]
predicted_advertising_costs = model.predict(sales_values)

# Menampilkan hasil prediksi
print("Prediksi biaya iklan untuk 50, 100, dan 150 sales:")
for sales, cost in zip([50, 100, 150], predicted_advertising_costs):
    print(f'{sales} sales: ${cost:.2f} juta')

# Menghitung RMSE dan R2 score
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

# Visualisasi
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Hubungan antara Penjualan dan Biaya Iklan')
plt.xlabel('Penjualan (million $)')
plt.ylabel('Biaya Iklan (million $)')
plt.show()
