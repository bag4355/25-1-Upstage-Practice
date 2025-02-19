import tensorflow as tf
from tensorflow import keras

# 1. MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 흑백 반전: 255에서 빼서 반전 (숫자가 검정 → 흰색, 배경이 흰색 → 검정)
x_train = 255 - x_train
x_test  = 255 - x_test

# 정규화
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# CNN 입력에 맞게 (28,28,1) 형태로 변환
x_train = x_train[..., None]
x_test  = x_test[..., None]

# 2. 데이터 증강 레이어 (학습 시에만 적용)
data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.1)
])

# 3. 개선된 간소화된 CNN 모델 정의 (데이터 증강 포함)
inputs = keras.layers.Input(shape=(28, 28, 1))
# 증강: 훈련시에만 적용되고, 추론시에는 자동 비활성화됨
x = data_augmentation(inputs)

# 첫 번째 컨볼루션 블록
x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.25)(x)

# 두 번째 컨볼루션 블록
x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.25)(x)

# 분류기
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model = keras.models.Model(inputs=inputs, outputs=outputs)

# 4. 컴파일 및 모델 체크포인트 콜백 설정
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_mnist_model.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# 5. 모델 학습
model.fit(x_train, y_train, epochs=35, batch_size=32, validation_split=0.2, callbacks=callbacks)

# 6. 최적 모델 불러와서 테스트 평가 및 최종 모델 저장
best_model = keras.models.load_model("best_mnist_model.h5")
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print("테스트 정확도:", test_acc)

best_model.save("mnist_model.h5")
print("모델이 저장되었습니다: mnist_model.h5")

