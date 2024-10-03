import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt

# CIFAR-10 데이터셋 로드 및 정규화
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 데이터 시각화
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# CNN 모델 구축
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(LeakyReLU(alpha=0.1))
model.add(layers.Dense(10))

model.summary()

# 모델 컴파일
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# 모델 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)

# 정확도 및 손실 시각화
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label = '검증 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label = '검증 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend(loc='upper right')
plt.show()

