import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터 생성 (가상의 시계열 데이터, 예시로 사용)
# 시계열 데이터(음성 데이터 대신 시뮬레이션된 데이터 사용)
def generate_time_series(n_samples, n_timesteps):
    X = np.array([np.sin(np.linspace(0, 3*np.pi, n_timesteps)) for _ in range(n_samples)])
    y = np.array([np.cos(np.linspace(0, 3*np.pi, n_timesteps)) for _ in range(n_samples)])
    return X, y

n_samples = 1000
n_timesteps = 50
X, y = generate_time_series(n_samples, n_timesteps)

# 2. 데이터 전처리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. LSTM 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='tanh', input_shape=(n_timesteps, 1)),  # LSTM 레이어
    tf.keras.layers.Dense(1)  # 출력 레이어 (회귀 문제, 음성 인식에선 Softmax 등 다르게 설정 가능)
])

# 4. 모델 컴파일 (손실 함수는 MSE, 최적화 알고리즘은 Adam)
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. 모델 학습
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # LSTM 입력을 3차원으로 변환
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 6. 모델 평가
test_loss = model.evaluate(X_test, y_test)
print(f"테스트 손실: {test_loss:.4f}")

# 7. 예측 결과 확인
predictions = model.predict(X_test)
print("첫 번째 테스트 샘플의 예측:", predictions[0])
