# modules
import streamlit as st

# content
st.set_page_config(page_title="HORESA - Streamlit", page_icon=":hotel:")

header = '<p style="font-family:Monotype Corsiva; color:Green; font-size: 50px; font-weight: bold;">HORESA</p>'
st.markdown(header, unsafe_allow_html=True)
st.subheader("Introduction")
st.write("I am a 3rd-year Data Science student studying in Universiti Malaya and this data product is one of the "
         "deliverables of WIH3001 Data Science Project.\n\n"
         "**HORESA** simply stands for Hotel Review Sentiment Analyzer. This is an web application for sentiment "
         "analysis on hotel reviews built on top of Python and integration of Streamlit. "
         "The model is trained several times with data scraped from Tripadvisor and additional reviews from Kaggle "
         "for upsampling purpose. Ultimately, the weighted F1-score is as high as 87%.")

st.subheader("User Manual")
st.write("**SINGLE TEXT**\n"
         "1. Type any reviews in the text area.\n"
         "2. Click \"Analyze\".\n"
         "3. View output (\"Labels\" and \"Confidence Level\").\n"
         "4. To predict new sentiment, replace with new text.")

st.write("**CSV FILE**\n"
         "1. Click \"Browse files\" to upload any reviews as .csv file format.\n"
         "2. View output (\"Labels\" and \"Confidence Level\").\n"
         "3. Below the table shows a donut chart on the distribution of labels.\n"
         "4. To download the table, click \"Export as .csv\".")

st.subheader("Contact Me")
st.write("If you encountered any problems, feel free to:\n"
        "- Email me - jylim0331@gmail.com\n"
        "- WhatsApp me - https://wa.me/+60108377110\n"
        "Here is my Github repo as well.\n"
        "- https://github.com/LimJingYao")