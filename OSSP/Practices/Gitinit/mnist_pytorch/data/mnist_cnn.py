import tensorflow as tf
from tensorflow import keras

# 1. 데이터 로드
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. 전처리
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. 모델 정의 (CNN)
model = keras.models.Sequential([
    keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

# 4. 컴파일 & 학습
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5)

# 5. 평가
model.evaluate(X_test, y_test)