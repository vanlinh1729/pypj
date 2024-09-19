import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Đọc file CSV
file_path = 'Advertising Budget and Sales.csv'
data = pd.read_csv(file_path)

# Xóa cột 'Unnamed: 0' nếu không cần thiết
data_cleaned = data.drop(columns=['Unnamed: 0'])

# Tách dữ liệu thành features (X) và target (y)
X = data_cleaned[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = data_cleaned['Sales ($)']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo các mô hình cơ bản
linear_reg = LinearRegression()
ridge_reg = Ridge()
neural_net = MLPRegressor(random_state=42, max_iter=2000)

# Khởi tạo mô hình Stacking với các mô hình cơ bản
stacking_model = StackingRegressor(
    estimators=[
        ('linear', linear_reg),
        ('ridge', ridge_reg),
        ('neural_net', neural_net)
    ],
    final_estimator=LinearRegression()
)

# Huấn luyện mô hình Stacking
stacking_model.fit(X_train_scaled, y_train)

# Dự đoán và tính toán lỗi trên tập test
y_pred = stacking_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

# Ghi nội dung vào file
with open('MSE.txt', 'w', encoding='utf-8') as file:
    file.write(str(mse))
#print("Mean Squared Error:", mse)

# Lưu mô hình và scaler vào file
joblib.dump(stacking_model, 'stacking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
