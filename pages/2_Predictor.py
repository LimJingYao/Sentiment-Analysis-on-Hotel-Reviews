# modules
import re
import nltk
import string
import pickle
import numpy as np
import contractions
import pandas as pd
import streamlit as st
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# load packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# load model and vectorizer
model = pickle.load(open('my_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# helper function
def cleaning(text):
    html_pattern = re.compile('<.*?>')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = html_pattern.sub(r'', text)
    text = url_pattern.sub(r'', text)
    text = re.sub("@\S+", "", text)
    text = text.encode('ascii', 'ignore').decode()
    text = contractions.fix(text)
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).lower()
    text = re.sub('\s{2,}', " ", text)
    text = re.sub("\n", " ", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")]) # remove stopwords
    text = ' '.join([WordNetLemmatizer().lemmatize(word, pos='v') for word in text.split()]) # lemmatization
    return text

def single_sentiment_predict(text):
    processed_text = cleaning(text)
    text_tfidf = tfidf.transform([processed_text])
    probability = model.predict_proba(text_tfidf)
    result = pd.DataFrame(columns=['Labels', 'Confidence Level'])
    result['Labels'] = model.classes_
    result['Confidence Level'].iloc[0] = probability.item(0)
    result['Confidence Level'].iloc[1] = probability.item(1)
    result['Confidence Level'].iloc[2] = probability.item(2)
    return result

def batch_sentiment_predict(text):
    processed_text = cleaning(text)
    text_tfidf = tfidf.transform([processed_text])
    probability = model.predict_proba(text_tfidf)
    reference = np.argmax(probability)
    pred = model.classes_[reference]
    probability = round(probability[0][reference], 4)
    return pred, probability

def color_label(label, color= '#00f900'): # lightgreen
    if label == 'Bad':
        color = 'red'
    elif label == 'OK':
        color = 'yellow'
    return f'background-color: {color}'

# content
st.set_page_config(page_title="HORESA - Streamlit", page_icon=":hotel:")

header = '<p style="font-family:Monotype Corsiva; color:Green; font-size: 50px; font-weight: bold;">HORESA</p>'
st.markdown(header, unsafe_allow_html=True)
st.subheader('Input Selection')
input = st.radio(
                "",
                ["Single Text", "CSV File"],
                horizontal=True,
                label_visibility="collapsed"
)

if input == 'Single Text':
    st.caption("_Note: Feel free to test with any reviews, but **ONLY** in **:red[ENGLISH]**_")
    st.subheader('Enter your review here:')
    text = st.text_area("", label_visibility="collapsed")
    predict = st.button("Analyze")

    if predict:
        if text:
            output, image = st.columns(2)
            with output:
                result = single_sentiment_predict(text)
                st.dataframe(result.style.highlight_max(subset = ['Confidence Level'],axis = 0),
                             use_container_width=True)
            with image:
                index = np.argmax(result["Confidence Level"])
                pred = result.iloc[index][0]
                if pred == 'Bad':
                    st.image("dont-like.png")
                elif pred == 'OK':
                    st.image("okay.png")
                else:
                    st.image("awesome.png")
                    st.balloons()
        else:
            st.write("No reviews entered :white_frowning_face:")

    with st.sidebar:
        st.markdown("Guides:")
        st.write("**SINGLE TEXT**\n"
                 "1. Type any reviews in the text area.\n"
                 "2. Click \"Analyze\".\n"
                 "3. View output (\"Labels\" and \"Confidence Level\").\n"
                 "4. To predict new sentiment, replace with new text.")
        st.caption("_Note: The higher the \"Confidence Level\", the higher the chance of predicting accurately._")

else:
    st.caption("_Note: Feel free to analyze with CSV files, but **ONLY** in **:red[ENGLISH]**_")
    st.subheader('Choose a file:')
    csv_file = st.file_uploader("", label_visibility="collapsed", type='csv')
    if csv_file is not None:
        csv_file = pd.read_csv(csv_file)
        pred, prob = zip(*csv_file[csv_file.columns[0]].apply(batch_sentiment_predict))
        csv_file['Labels'] = pred
        csv_file['Confidence Level'] = prob
        st.dataframe(csv_file.style.applymap(color_label, subset=['Labels']),
                     use_container_width=True)

        st.download_button(
            "Export as .csv",
            csv_file.to_csv(),
            "analyzed_text.csv"
        )

        data_plot = csv_file.groupby('Labels').count().reset_index()
        fig = px.pie(data_plot, names='Labels', values=csv_file.columns[0], width=450, height=400, hole=.6)
        fig.update_traces(textinfo='percent+label', marker=dict(colors=['#FFC0CB', '#98FB98', '	#FFEFD5'])) # pink, palegreen, papayawhip
        fig.update(layout_showlegend=False)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with st.sidebar:
        st.markdown("Guides:")
        st.write("**CSV FILE**\n"
                 "1. Click \"Browse files\" to upload any reviews as .csv file format.\n"
                 "2. View output (\"Labels\" and \"Confidence Level\").\n"
                 "3. Below the table shows a donut chart on the distribution of labels.\n"
                 "4. To download the table, click \"Export as .csv\".")
        st.caption("_Note: Make sure the first column in the CSV file is the reviews_")
