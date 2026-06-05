import pandas as pd
import streamlit as st
from PIL import Image
from model import load_model_and_predict  # только эта функция нужна


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Titanic",
        page_icon="🚢",
    )

    st.write(
        """
        # 🚢 Классификация пассажиров Титаника
        Определяем, кто из пассажиров выживет, а кто – нет.
        """
    )

    try:
        image = Image.open('titanic.jpg')
        st.image(image)
    except:
        st.info("Изображение не загружено")


def write_user_data(df):
    st.write("## 📊 Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## 🎯 Предсказание")
    st.success(prediction)

    st.write("## 📈 Вероятность")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    # Предсказание (используем только load_model_and_predict)
    prediction, prediction_probas = load_model_and_predict(user_input_df)

    write_user_data(user_input_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    embarked = st.sidebar.selectbox("Порт посадки", ("Шербур-Октевиль", "Квинстаун", "Саутгемптон"))
    pclass = st.sidebar.selectbox("Класс", ("Первый", "Второй", "Третий"))
    age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20, step=1)
    sib_sp = st.sidebar.slider("Братьев / сестер / супругов на борту", min_value=0, max_value=10, value=0, step=1)
    par_ch = st.sidebar.slider("Детей / родителей на борту", min_value=0, max_value=10, value=0, step=1)

    translation = {
        "Мужской": "male",
        "Женский": "female",
        "Шербур-Октевиль": "C",
        "Квинстаун": "Q",
        "Саутгемптон": "S",
        "Первый": 1,
        "Второй": 2,
        "Третий": 3,
    }

    data = {
        "Pclass": translation[pclass],
        "Sex": translation[sex],
        "Age": age,
        "SibSp": sib_sp,
        "Parch": par_ch,
        "Embarked": translation[embarked]
    }

    return pd.DataFrame(data, index=[0])


if __name__ == "__main__":
    process_main_page()
