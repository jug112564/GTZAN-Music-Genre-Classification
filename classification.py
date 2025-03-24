import os
import librosa
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

X = []
Y = []
song_type = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
             'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
fixed_mfcc_shape = (40, 130)

#preprocessing data
for genre, label in song_type.items():
    folder_path = f'./Data/genres_original/{genre}'
    for file_name in os.listdir(folder_path):
        print('Preprocessing audio data (' + file_name + ')')
        file_path = os.path.join(folder_path, file_name)
        audio, sr = librosa.load(file_path)

        # data augmentation
        audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)

        # MFCC
        for data in [audio, audio_pitch]:
            mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=fixed_mfcc_shape[0])

            # MFCC padding
            if mfcc.shape[1] < fixed_mfcc_shape[1]:
                pad_width = fixed_mfcc_shape[1] - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :fixed_mfcc_shape[1]]

            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
            mfcc = np.expand_dims(mfcc, axis=-1)

            X.append(mfcc)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# CNN model
model = Sequential()
model.add(ZeroPadding2D(padding=2, input_shape=(fixed_mfcc_shape[0], fixed_mfcc_shape[1], 1)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation="softmax"))

# model compile
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)