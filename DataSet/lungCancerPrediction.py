# Import thư viện cho đại số tuyến tính và các phép tính số học
import numpy as np 
# Import thư viện xử lý dữ liệu dạng bảng và đọc file CSV
import pandas as pd 
# Import hàm chia dữ liệu thành tập huấn luyện và kiểm tra
from sklearn.model_selection import train_test_split
# Import thuật toán Random Forest cho phân loại
from sklearn.ensemble import RandomForestClassifier
# Import bộ mã hóa nhãn để chuyển dữ liệu phân loại sang dạng số
from sklearn.preprocessing import LabelEncoder
# Import hàm tính điểm F1 để đánh giá mô hình
from sklearn.metrics import f1_score

# ---------------------------------------------------
# Bước 1: Tải dữ liệu (hiện đang bị comment)
# ---------------------------------------------------

df = pd.read_csv(
    r"C:/Users/Admin/.cache/kagglehub/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/versions/2/cancer patient data sets.csv")
# Cần bỏ comment dòng trên để tải dữ liệu bệnh nhân ung thư từ file CSV

# ---------------------------------------------------
# Bước 2: Chuẩn bị đặc trưng và nhãn mục tiêu
# ---------------------------------------------------
# Khởi tạo LabelEncoder để chuyển đổi nhãn phân loại thành dạng số
le = LabelEncoder()

# Mã hóa cột 'Level' (biến mục tiêu) thành dạng số (ví dụ: 'Ác tính'=1, 'Lành tính'=0)
df["Level"] = le.fit_transform(df["Level"])

# Loại bỏ cột 'Patient Id' và 'index' vì không hữu ích cho dự đoán
df = df.drop(["Patient Id", "index"], axis=1)

# ---------------------------------------------------
# Bước 3: Tách đặc trưng và biến mục tiêu
# ---------------------------------------------------

# Tạo tập đặc trưng 'X' bằng cách bỏ cột 'Level' khỏi DataFrame
X = df.drop("Level", axis=1)
# Định nghĩa biến mục tiêu 'y' là cột 'Level' (giá trị cần dự đoán)
y = df["Level"]

# ---------------------------------------------------
# Bước 4: Chia dữ liệu thành tập huấn luyện và kiểm tra
# ---------------------------------------------------

# Chia dữ liệu theo tỷ lệ 80-20 (20% cho kiểm tra, 80% cho huấn luyện)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# Bước 5: Khởi tạo và huấn luyện mô hình Random Forest
# ---------------------------------------------------


# Khởi tạo mô hình Random Forest với 100 cây quyết định
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện mô hình Random Forest với dữ liệu huấn luyện
rf_model.fit(X_train, y_train)

# ---------------------------------------------------
# Bước 6: Dự đoán và đánh giá mô hình
# ---------------------------------------------------

# Sử dụng mô hình đã huấn luyện để dự đoán trên tập kiểm tra
y_pred = rf_model.predict(X_test)

# Tính điểm F1 có trọng số để đánh giá hiệu suất mô hình
f1 = f1_score(y_test, y_pred, average="weighted")

# In điểm F1 có trọng số làm tròn đến 4 chữ số thập phân
print(f"Weighted F1 Score: {f1:.4f}")

# ---------------------------------------------------
# Bước 7: Phân tích tầm quan trọng của các đặc trưng
# ---------------------------------------------------

# Tạo DataFrame lưu trữ độ quan trọng của từng đặc trưng
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)  # Sắp xếp theo độ quan trọng giảm dần

# In 5 đặc trưng quan trọng nhất
print("\nTop 5 most important features:")
print(feature_importance.head())