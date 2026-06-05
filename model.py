# model.py — версия для деплоя
import pandas as pd
from pickle import load


def preprocess_data(df: pd.DataFrame):
    """Предобработка для предсказания (без обучения)"""
    to_encode = ['Sex', 'Embarked']

    for col in to_encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        df.drop(col, axis=1, inplace=True)

    # Добавляем недостающие колонки
    expected_cols = ['Pclass', 'Age', 'SibSp', 'Parch',
                     'Sex_female', 'Sex_male',
                     'Embarked_C', 'Embarked_Q', 'Embarked_S']

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_cols]


def load_model_and_predict(df, path="model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    # Предобработка
    df_processed = preprocess_data(df.copy())

    # Предсказание
    prediction = model.predict(df_processed)[0]
    prediction_proba = model.predict_proba(df_processed)[0]

    # Красивый вывод
    encode_prediction = {
        0: "Сожалеем, вам не повезло ❌",
        1: "Ура! Вы выживете! ✅"
    }

    result_df = pd.DataFrame({
        "Вам не повезло с вероятностью": [prediction_proba[0]],
        "Вы выживете с вероятностью": [prediction_proba[1]]
    })

    return encode_prediction[prediction], result_df