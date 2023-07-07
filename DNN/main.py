from sklearn.model_selection import train_test_split
from preprocessing import *
from models import get_lstm, train
from utils import visualize

if __name__ == '__main__':
    # 데이터 전처리 및 증강
    df_augmented, encoder = preprocess()

    # 토크나이즈 및 패드 시퀀스 생성
    data_pad, tokenizer = tokenize(df_augmented)

    # 데이터셋 분리
    X_train, X_val, y_train, y_val \
        = train_test_split(data_pad, df_augmented['class'], test_size=0.1, random_state=42)

    # 모델 생성 및 학습
    model, history = train(
        get_lstm,
        df_augmented,
        tokenizer,
        data_pad,
        X_train,
        y_train,
        X_val,
        y_val
    )

    # 시각화
    visualize(history)


