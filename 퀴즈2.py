import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('/content/한국_기업문화_HR_데이터셋_샘플.csv')

print(df.isnull().sum())

#이직여부를 이진값으로 변환
df['이직여부'] = df['이직여부'].map({'Yes': 1, 'No': 0})

df.drop(columns=['Over18', 'EmployeeCount', 'EmployeeNumber', 'StandardHours'], inplace=True)

one_hot_cols = ['출장빈도', '부서', '전공분야', '직무', '결혼상태']
label_cols = ['성별', '야근여부']

# One-Hot Encoding 순서 x
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

# Label Encoding 
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

print(df)
selected_features = ['Age', '근무환경만족도', '집까지거리', '월급여', '업무만족도'] 
#어릴수록 이직 확률 up, 만족도가 낮으면 이직 확률 up , 통근시간이 길 수록 삶의 질 하락 이직 확률 up
#월급여,업무만족도 역시 이직 확률에 큰 영향을 미칠 것. 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = df[selected_features]
y = df['이직여부']

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
