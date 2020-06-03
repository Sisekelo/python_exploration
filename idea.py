import streamlit as st
import os
import pandas as pd
import validators as validate
import seaborn as sns
from matplotlib import pyplot as plt
from bokeh.models.widgets import Div

st.title("Data Analyser!")
st.subheader("This is a data analyser allwoing users to simply input their file and get some initial information about that data without any code")
st.write("The aim is to remove all redundant initial exercises that most data analysts do before actually using ML algo's on it.")
st.write("reach out: *sidlamini15@alustudent.com*")


def main():
    #chooose method of input
    choose_model = st.selectbox("How do you want to input data?", ["Link", "Upload"])

    if(choose_model == "Upload"):
        waiting = st.write("")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            showData(data)
        else:
            waiting = st.write("Waiting for upload")

    else:
        st.info("example: *https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data*")
        text = st.text_input('Input URL', '')

        if len(text) is not 0:
            waiting = st.write("")
            data = pd.read_csv(text, header=None)

            showData(data)
        else:
            waiting = st.write("Waiting for upload")


def showData(data):
    if st.checkbox('See head of the data'):
        st.write(data.head())
    if st.checkbox("General info"):
        st.write(data.describe())
    if st.checkbox("Missing values"):
        missing_data = getMissingValues(data)
        st.write(missing_data)
    if st.checkbox("Correlation info"):
        st.write("*best for 10 or less features*")
        corrmat = data.corr()
        fig = plt.figure(figsize = (12, 9))
        sns.heatmap(corrmat, vmax = 1, square = True,annot=True)
        plt.show()
        st.pyplot()

    if st.button('Buy us coffee☕️'):
        js = "window.open('https://www.buymeacoffee.com/Sisekelo')"  # New tab or window
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

def getMissingValues(data):
    #show me all missing values
    total = data.isnull().sum().sort_values(ascending=False)

    #percentage of missing
    percent_1 = data.isnull().sum()/data.isnull().count()*100

    #round off to the nearest one, sort by highest
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

    #concatinate the array
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

    if missing_data.Total.unique().any() == 0:
        missing_data = "No missing data!😎"
    else:
        missing_data = missing_data

    return missing_data

if __name__ == "__main__":
	main()
