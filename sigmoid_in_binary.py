import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# 1. 데이터 생성 (이진 분류 문제)
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# 2. 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 데이터 정규화 (StandardScaler 사용)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # 입력 레이어
    tf.keras.layers.Dense(16, activation='relu'),     # 은닉층 1
    tf.keras.layers.Dense(8, activation='relu'),      # 은닉층 2
    # 이진 분류를 위한 출력층, 여기에 sigmoid 함수가 적용됩니다.
    tf.keras.layers.Dense(1, activation='sigmoid')    # 출력층 (sigmoid 사용)
])

# 5. 모델 컴파일 (손실 함수는 binary_crossentropy, 최적화 알고리즘은 Adam 사용)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. 모델 학습
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 7. 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {test_accuracy:.4f}")

# 8. 예측과 확률적 해석
predictions = model.predict(X_test)

# 첫 번째 샘플의 예측 확률과 클래스를 확인
for i in range(5):  # 처음 5개의 샘플만 출력
    print(f"샘플 {i+1}의 예측 확률: {predictions[i][0]:.4f}")
    predicted_class = 1 if predictions[i][0] >= 0.5 else 0
    print(f"샘플 {i+1}의 예측 클래스: {predicted_class}")
