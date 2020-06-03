import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import facebook
import json
import dateutil.parser as dateparser
from pylab import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from PIL import Image
import re
import datetime


#  here is token which you get from Facebook Graph APIs, every time using program, you need update this token
token = "EAArdmq4wewsBAPZBCRYrtZCH4ZAhxt0h85xL6RvQnfF5BYTEupWMTRHiEhBacAuIGiZBd3DqVIBfpM7UBeJR0LrjHUHEJxLu7nZCZCjGEFtUvasJh0XDlZBZAZBrSGSmbwZAThOsNQyEhZBMHFZBUdYF7ak5rBSrP7D3cU9FYcsuKES4JSVGDAo203pfudBqX5563xFRNRIBS8YhOxWUrGWBlOnufcyd4cPQDyG6Sd1OzgddVQZDZD"

graph = facebook.GraphAPI(access_token=token, version="2.12")

page_id = '108419247500492'

# Get the message from a post.
posts = graph.get_object(id=page_id, fields='feed')

title = graph.get_object(id=page_id, fields='')

# setting the x - coordinates
dates = []
# setting the corresponding y - coordinates
polarities = []

messages = []

st.title("Welcome to the sentiment analysis of: "+ title['name'])
st.write("Get a sense of how people feel about your brand.")
image = Image.open('Black_Analytics.jpg')

st.image(image, caption='Doing analytics',use_column_width=True)

def getSentiment(blob):
    blob_output = []

    for x in blob:

        try:
            message = x['message']
        except:
          message = ""

        date = x['created_time']
        id = x['id']

        regex = re.compile('[^a-zA-Z]')

        text = regex.sub('', message)

        blob_message = TextBlob(message)
        polarity = blob_message.sentiment.polarity
        subjectivity = blob_message.sentiment.subjectivity

        dates.append(date)
        polarities.append(polarity)
        messages.append(message)

        blob_output.append([id,date,polarity,message])

    return blob_output

home_sentiment = getSentiment(posts['feed']['data'])

st.title("Sentiment Analysis of "+title['name'])
df = pd.DataFrame(
     home_sentiment,
     columns=['id','date','sentiment','message'])

st.vega_lite_chart(df, {
    "width": 700,
    "height": 400,
     'mark': {'type': 'line', 'point': True,'tooltip': {"content":posts['feed']['data']}},
     'encoding': {
         'y': {'field': 'sentiment', 'type': 'quantitative'},
         'x': {'field': 'date', 'type': 'temporal'}
     },
})

st.write('Analysis of comments from specific post')

#drop down menu
option = st.selectbox('Select which post you want to see:',(messages))

for x in home_sentiment:
    message = x[3]
    if message is option:
        post_id = x[0]
        comments = graph.get_object(id=post_id, fields='comments')
        #st.write(comments)

        if 'comments' in comments:
            comment_sentiment = getSentiment(comments['comments']['data'])

            df2 = pd.DataFrame(
                 comment_sentiment,
                 columns=['id','date','sentiment','message'])

            st.vega_lite_chart(df2, {
                "width": 700,
                "height": 400,
                 'mark': {'type': 'line', 'point': True,'tooltip': {"content":comments['comments']['data']}},
                 'encoding': {
                     'y': {'field': 'sentiment', 'type': 'quantitative'},
                     'x': {'field': 'date', 'type': 'temporal'}
                 },
            })
        else:
            st.write('Looks like you dont have any comments for this post.😅')
        break
