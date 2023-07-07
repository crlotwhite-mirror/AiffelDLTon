from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM, Dropout, Conv1D, GlobalMaxPooling1D


def get_mlp(data, tokenizer, data_pad):
    # MLP 모델 생성
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=data_pad.shape[1]))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(data['class'].unique()), activation='softmax'))
    return model


def get_cnn(data, tokenizer, data_pad):
    # CNN 모델 생성
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=data_pad.shape[1]))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(data['class'].unique()), activation='softmax'))

    return model


def get_lstm(data, tokenizer, data_pad):
    # LSTM 모델 생성
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=data_pad.shape[1]))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(data['class'].unique()), activation='softmax'))

    return model


def train(get_model, data, tokenizer, data_pad, X_train, y_train, X_val, y_val):
    model = get_model(data, tokenizer, data_pad)
    model.summary()
    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 모델 학습
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
    return model, history
