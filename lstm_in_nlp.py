import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 샘플 텍스트 데이터
texts = [
    "I love natural language processing",
    "LSTM models are very powerful",
    "I enjoy learning about deep learning",
    "NLP tasks are interesting"
]

labels = [1, 1, 1, 0]  # 예시로 텍스트를 두 클래스로 분류

# 1. 텍스트 전처리 (토크나이저 사용)
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print(f"단어 인덱스: {word_index}")

# 패딩 처리 (시퀀스 길이 맞추기)
data = pad_sequences(sequences, maxlen=10)

# 2. LSTM 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=100, output_dim=16, input_length=10),  # 임베딩 레이어
    tf.keras.layers.LSTM(32),  # LSTM 레이어
    tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류를 위한 출력층
])

# 3. 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, np.array(labels), epochs=10, batch_size=2)

# 4. 새로운 텍스트 예측
new_texts = ["I love learning about NLP", "This task is difficult"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_data = pad_sequences(new_sequences, maxlen=10)

predictions = model.predict(new_data)
print(f"예측 결과: {predictions}")
