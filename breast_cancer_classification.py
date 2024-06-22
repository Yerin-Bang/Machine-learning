import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# 데이터 불러오기
file_path = r"C:\Users\HighTech\Desktop\breast-cancer-wisconsin-data.csv"
df = pd.read_csv(file_path)

# 데이터의 첫 5개 행과 열 이름 확인
print(df.head())
print(df.columns)

# 특성과 타겟 변수 분리
X = df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1, errors='ignore')  # 'Unnamed: 32'가 없으면 무시
y = df['diagnosis'].map({'M': 1, 'B': 0})  # 악성(M)을 1로, 양성(B)을 0으로 매핑

# 훈련 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 초기화 및 학습
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 분류 보고서
print(classification_report(y_test, y_pred))

# 혼동 행렬
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 교차 검증
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean():.2f}')