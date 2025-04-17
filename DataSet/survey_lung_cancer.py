import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Đọc dữ liệu
path = "F:/ML/DataSet/survey_lung_cancer.csv"
data = pd.read_csv(path)

# In thông tin của dữ liệu
# print("Thông tin dữ liệu:")
# print(data.info())
# print("\nMẫu dữ liệu:")
# print(data.head())

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
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': clf.feature_importances_
# }).sort_values('Importance', ascending=False)
#
# print("\nTầm quan trọng của các đặc trưng:")
# print(feature_importance.head(10))
# Tạo và huấn luyện các mô hình khác nhau
model_classes = {
    'LR': LogisticRegression,
    'DT': DecisionTreeClassifier,
    'RF': RandomForestClassifier,
    'SVM': SVC,
    'KNN': KNeighborsClassifier,
    'NV': GaussianNB
}

# Đánh giá độ chính xác của từng mô hình
precision_scores = {}
for name, model_class in model_classes.items():
    if name == 'LR':
        model = model_class(max_iter=1000)
    elif name == 'RF':
        model = model_class(n_estimators=100)
    else:
        model = model_class()
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    precision_scores[name] = round(precision, 2)
    print(f"{name} Precision: {precision:.2f}%")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
names = list(precision_scores.keys())
scores = list(precision_scores.values())
plt.plot(names, scores, marker='o', linestyle='-', linewidth=2, markersize=8)

# Thêm nhãn cho từng điểm
for i, score in enumerate(scores):
    plt.annotate(f"{score}", (names[i], score), textcoords="offset points", 
                 xytext=(0,10), ha='center')

plt.xlabel('Classifier Techniques')
plt.ylabel('Precision')
plt.title('Classifier Techniques vs Precision')
plt.ylim(min(scores)-1, max(scores) + 1)  # Điều chỉnh giới hạn trục y linh hoạt
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('classifier_precision_comparison.png', dpi=300)

# Hiển thị biểu đồ
plt.show()
