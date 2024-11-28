import pandas as pd
import streamlit as st

st.title("Анализ выживших женщин на Титанике")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/AfanasevAndU/titanik_train_data/main/titanic_train%20%5BtsKg9Q%5D.csv'
    return pd.read_csv(url)

df = load_data()

survived_women = df[(df['Sex'] == 'female') & (df['Survived'] == 1)]

result = survived_women.groupby('Pclass').agg(
    Count=('PassengerId', 'count'),
    Fare_Range_Min=('Fare', 'min'),
    Fare_Range_Max=('Fare', 'max')
).reset_index()

st.header("Результаты анализа")
st.dataframe(result)

csv_data = result.to_csv(index=False)
st.download_button(
    label="Скачать результаты в формате CSV",
    data=csv_data,
    file_name="survived_women_analysis.csv",
    mime="text/csv"
)

st.sidebar.header("Настройки фильтрации")
selected_classes = st.sidebar.multiselect(
    "Выберите классы обслуживания для отображения:",
    options=result["Pclass"].unique(),
    default=result["Pclass"].unique()
)

filtered_result = result[result["Pclass"].isin(selected_classes)]

st.header("Фильтрованные данные")
st.dataframe(filtered_result)
