import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,HuberRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

#read data
data=pd.read_csv("StudentScore.xls")

#split data
#vertical
target="writing score"
lst_drop=[target]
x=data.drop(lst_drop,axis=1)
y=data[target]
#horizontal
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# #data processing
#numical
num_processing=Pipeline(steps=[
    ("inpute", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

#ordinal
ord_parental=['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
gender_values=x_train["gender"].unique()
lunch_values=x_train["lunch"].unique()
test_values=x_train["test preparation course"].unique()

ord_processing=Pipeline(steps=[
    ("inpute", SimpleImputer(strategy="most_frequent")),
    ("encoder",OrdinalEncoder(categories=[ord_parental,gender_values,lunch_values,test_values]))
])

#nominal
nom_processing=Pipeline(steps=[
    ("inpute", SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder())
])
#compose processing data
preprocessor=ColumnTransformer([
    ("num_feature",num_processing,['reading score','math score']),
    ("ord_feature",ord_processing,['parental level of education','gender','lunch','test preparation course']),
    ("nom_feature",nom_processing,['race/ethnicity'])
])


# train model
reg=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",LinearRegression())
])
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

# print(mean_absolute_error(y_test,y_predict)

# ========== GUI for Predict ========== #

def predict_from_input():
    try:
        # Lấy dữ liệu từ các ô nhập
        reading = float(entry_reading.get())
        math = float(entry_math.get())
        parental = combo_parental.get()
        gender = combo_gender.get()
        lunch = combo_lunch.get()
        test_prep = combo_test.get()
        race = combo_race.get()
        # Tạo DataFrame mẫu
        input_df = pd.DataFrame({
            'reading score': [reading],
            'math score': [math],
            'parental level of education': [parental],
            'gender': [gender],
            'lunch': [lunch],
            'test preparation course': [test_prep],
            'race/ethnicity': [race]
        })
        # Dự đoán
        y_pred = reg.predict(input_df)[0]
        label_result.config(text=f"Kết quả dự đoán: {y_pred:.2f}")
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))

# Lấy các giá trị unique cho combobox
parental_list = list(ord_parental)
gender_list = list(gender_values)
lunch_list = list(lunch_values)
test_list = list(test_values)
race_list = list(x_train['race/ethnicity'].unique())

font_label = ("Arial", 12)
font_entry = ("Arial", 12)
font_button = ("Arial", 12, "bold")
font_result = ("Arial", 12, "bold")

root = tk.Tk()
root.title("Dự đoán điểm writing score")
root.geometry("420x420")

# Các nhãn và ô nhập
row = 0
tk.Label(root, text="Reading Score:", font=font_label).grid(row=row, column=0, sticky='e')
entry_reading = tk.Entry(root, width=22, font=font_entry)
entry_reading.grid(row=row, column=1)
row += 1

tk.Label(root, text="Math Score:", font=font_label).grid(row=row, column=0, sticky='e')
entry_math = tk.Entry(root, width=22, font=font_entry)
entry_math.grid(row=row, column=1)
row += 1

tk.Label(root, text="Parental Level of Education:", font=font_label).grid(row=row, column=0, sticky='e')
combo_parental = ttk.Combobox(root, values=parental_list, state='readonly', width=20, font=font_entry)
combo_parental.grid(row=row, column=1)
combo_parental.current(0)
row += 1

tk.Label(root, text="Gender:", font=font_label).grid(row=row, column=0, sticky='e')
combo_gender = ttk.Combobox(root, values=gender_list, state='readonly', width=20, font=font_entry)
combo_gender.grid(row=row, column=1)
combo_gender.current(0)
row += 1

tk.Label(root, text="Lunch:", font=font_label).grid(row=row, column=0, sticky='e')
combo_lunch = ttk.Combobox(root, values=lunch_list, state='readonly', width=20, font=font_entry)
combo_lunch.grid(row=row, column=1)
combo_lunch.current(0)
row += 1

tk.Label(root, text="Test Preparation Course:", font=font_label).grid(row=row, column=0, sticky='e')
combo_test = ttk.Combobox(root, values=test_list, state='readonly', width=20, font=font_entry)
combo_test.grid(row=row, column=1)
combo_test.current(0)
row += 1

tk.Label(root, text="Race/Ethnicity:", font=font_label).grid(row=row, column=0, sticky='e')
combo_race = ttk.Combobox(root, values=race_list, state='readonly', width=20, font=font_entry)
combo_race.grid(row=row, column=1)
combo_race.current(0)
row += 1

btn_predict = tk.Button(root, text="Dự đoán", command=predict_from_input, font=font_button)
btn_predict.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

label_result = tk.Label(root, text="Kết quả dự đoán: ", font=font_result)
label_result.grid(row=row, column=0, columnspan=2)

root.mainloop()