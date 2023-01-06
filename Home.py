# modules
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer

# for wordcloud
processed_data = pd.read_csv("https://raw.githubusercontent.com/LimJingYao/Sentiment-Analysis-on-Hotel-Reviews/master/data/cleaned_data.csv")
processed_data = processed_data.drop(columns=['Unnamed: 0'])
processed_data = processed_data.dropna(axis=0)

def WordCloudGenerator(data):
    wordCloud = WordCloud(max_words=1000, min_font_size=20, max_font_size=120, background_color='white',
                      height=600,width=1200, random_state=1).generate(' '.join(data))
    fig, ax = plt.subplots()
    ax.imshow(wordCloud)
    plt.axis("off")
    st.pyplot(fig)

def top15_ngram(text):
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_df=0.8).fit(text)
    sum_of_words = vectorizer.transform(text).sum(axis=0)
    words_freq = [(word, sum_of_words[0, index]) for word, index in vectorizer.vocabulary_.items()] # compile data into new variable
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True) # sort based on second column (the sum of words)
    return words_freq[:15]

def NGramGenerator():
    ngram = processed_data.copy()
    labels = ['Bad', 'OK', 'Good']
    colors = ['teal', 'lime', 'gold']

    fig = make_subplots(rows=3, cols=1) # create subplots
    for i in range(3):
        top15_trigrams = top15_ngram(ngram[ngram['Ratings'] == labels[i]]['Reviews'])[:15] #create N-grams for each label
        x, y = map(list, zip(*top15_trigrams)) # map data onto x and y
        fig.add_trace(go.Bar(x=y, y=x, orientation='h', type="bar", name=labels[i], marker=dict(color=colors[i])), i+1, 1) # barplot

    fig.update_layout(width=660,height=1000,title=dict(text='<b>Top 15 Trigrams for Each Rating Labels</b>', x=0.5, y=0.95))
    st.plotly_chart(fig)

def Length_Ratings():
    scraped_data = pd.read_csv("https://raw.githubusercontent.com/LimJingYao/Sentiment-Analysis-on-Hotel-Reviews/master/data/data.csv")
    scraped_data = scraped_data.rename(columns={'review': 'Reviews', 'bubble_rating': 'Ratings'})
    scraped_data['Ratings'] = scraped_data['Ratings'].replace(
        ['ui_bubble_rating bubble_10', 'ui_bubble_rating bubble_20', 'ui_bubble_rating bubble_30',
         'ui_bubble_rating bubble_40', 'ui_bubble_rating bubble_50'],
        ['1', '2', '3', '4', '5'])
    scraped_data['Length'] = scraped_data['Reviews'].apply(len)
    fig = px.strip(scraped_data, x='Ratings', y='Length', color='Ratings',
                   category_orders={'Ratings': ['1', '2', '3', '4', '5']})
    st.plotly_chart(fig)

# contents
st.set_page_config(page_title="HORESA - Streamlit", page_icon=":hotel:")

header = '<p style="font-family:Monotype Corsiva; color:Green; font-size: 50px; font-weight: bold;">HORESA</p>'
st.markdown(header, unsafe_allow_html=True)

st.write("Nowadays, making travel arrangements and bookings online is one of the most crucial commercial uses on the "
         "internet. Tourists and travellers are generating reviews, ratings and comments regarding their travelling "
         "experiences and feelings which could be an important source of information for company to analyse. "
         "Comparing to printed hotel brochure, reviews from customers often reflect the actual guest experiences, "
         "and are not influenced by marketing purposes. There are several ways to visualize the reviews:")

st.write("- Word Cloud")
st.caption("_This is an example of word cloud showing words which usually can be seen in good reviews._")
WordCloudGenerator(processed_data[processed_data['Ratings']=='Good'].Reviews)

st.write("- N-grams")
st.caption("_These are examples of trigrams showing words which usually can be seen in each type of reviews._")
NGramGenerator()

st.write("- Others")
st.caption("_We can also visualize the length of sentence for each type of labels._")
Length_Ratings()

link="""<a href="https://limjingyao-sentiment-analysis-on-hotel-reviews-home-7t0775.streamlit.app/Predictor" target = "_self">Predictor</a>"""
st.write("To better understand customers better, it is crucial to find out whether a review is positive, neutral "
         "or negative to the business. To predict their labels, try it out on the next page - "+link,
         unsafe_allow_html=True)


