def preprocess():
    import pandas as pd
    from koeda import EDA
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    # 데이터 불러오기
    data = pd.read_csv('train.csv')

    # class 열을 숫자로 변환
    encoder = LabelEncoder()
    data['class'] = encoder.fit_transform(data['class'])

    eda = EDA(
        morpheme_analyzer="Okt", alpha_sr=0.0, alpha_ri=0.3, alpha_rs=0.3, prob_rd=0.3
    )

    def run_eda(text):
        return eda(text, p=(0.9, 0.9, 0.9, 0.9), repetition=1)

    # 랜덤하게 행 선택 (예: 전체 행의 20%를 선택)
    random_indices = np.random.choice(data.index, size=int(len(data) * 0.3), replace=False)

    # 선택된 행에 대해 Random swap 함수 적용
    augmented_rows = data.loc[random_indices, 'conversation'].apply(run_eda)

    # 증강된 데이터를 복사하고, 'text' 열에 증강된 텍스트를 삽입
    new_rows = data.loc[random_indices].copy()
    new_rows['conversation'] = augmented_rows

    # 원본 데이터프레임에 증강된 데이터 추가
    df_augmented = pd.concat([data, new_rows])

    return df_augmented, encoder


def tokenize(data):
    from keras.preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences

    # 케라스를 이용한 정수 인코딩
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['conversation'])

    sequences = tokenizer.texts_to_sequences(data['conversation'])

    # 입력 데이터 패딩 처리
    data_pad = pad_sequences(sequences, padding='pre')
    return data_pad, tokenizer