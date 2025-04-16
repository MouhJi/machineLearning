import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Đọc dữ liệu
path = "F:/ML/DataSet/survey_lung_cancer.csv"
data = pd.read_csv(path)

# In thông tin của dữ liệu
print("Thông tin dữ liệu:")
print(data.info())
print("\nMẫu dữ liệu:")
print(data.head())

# Mã hóa biến mục tiêu LUNG_CANCER
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Chia dữ liệu thành các loại khác nhau để xử lý riêng
# 1. Cột số: AGE
numeric_cols = ['AGE']
# 2. Cột phân loại không phải boolean: GENDER 
categorical_cols = ['GENDER']
# 3. Cột boolean: tất cả các cột còn lại trừ 'LUNG_CANCER'
boolean_cols = [col for col in data.columns if col not in numeric_cols + categorical_cols + ['LUNG_CANCER']]

# Tạo DataFrame mới chỉ chứa các cột đã được xử lý
processed_data = pd.DataFrame()

# 1. Chuẩn hóa cột số
scaler = StandardScaler()
processed_data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 2. Mã hóa One-Hot cho cột phân loại
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = encoder.fit_transform(data[categorical_cols])
encoded_cols = [f"{col}_{cat}" for col, cats in zip(categorical_cols, encoder.categories_) for cat in cats[1:]]
processed_data[encoded_cols] = encoded_cats

# 3. Giữ nguyên các cột boolean hoặc đảm bảo chúng là kiểu số
for col in boolean_cols:
    processed_data[col] = data[col].astype(int)

# Chia dữ liệu
X = processed_data
y = data['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện mô hình
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = clf.predict(X_test)
print("\nKết quả đánh giá mô hình:")
print(classification_report(y_test, y_pred))

# Hiển thị tầm quan trọng của các đặc trưng
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTầm quan trọng của các đặc trưng:")
print(feature_importance.head(10))










