import streamlit as st
import joblib
from io import StringIO
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')


st.set_page_config(page_title='EMail Importance',layout='wide',)

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

path_kmeans = r"C:\Users\aredd\Desktop\MailImportance-ProjecctZ\Models\model_new.model"
path_pca = r"C:\Users\aredd\Desktop\MailImportance-ProjecctZ\Models\pca_model.model"
path_w2v = r"C:\Users\aredd\Desktop\MailImportance-ProjecctZ\Models\word2vec_model.model"

model_new = joblib.load(path_kmeans)
model_pca = joblib.load(path_pca)
model_w2v= Word2Vec.load(path_w2v)

cluster_centriods_pca = model_pca.fit_transform(model_new.cluster_centers_)

stop_words = list(set(stopwords.words('english')))

def process_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    text = [x for x in word_tokenize(text) if x not in stop_words and len(x) > 2]
    
    return text

def get_embeddings(tokens):
    embeddings = [model_w2v.wv[word] for word in tokens if word in model_w2v.wv]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return None


def return_sorted(emails):
    ret_lst = []
    for i in emails:
        test = process_text(i)
        test = get_embeddings(test)
    
        try:

            label = model_new.predict(test.reshape(1,100))
            test_pca = model_pca.transform(test.reshape(1,100))
            val = test_pca @ cluster_centriods_pca[label[0]]
            ret_lst.append([val,i])
        except Exception as E:
            st.write(E)
            st.warning("Ignorable Error")
    
    return ret_lst



st.title("Sort Your Mails")
st.markdown("<hr>",unsafe_allow_html=True)
st.markdown("""
<h2 style="color:green"> File Format *txt</h2>

<p style="color:green"> Emails should be</p>
<p style="color:green"> Email <br>
    Your email1.................... <br>
    <br>
    Email <br>
    Your email2.................... <br>
    <br>
    and so on......................
""",unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    string_data = uploaded_file.read().decode('utf-8')  
    emails = string_data.split('Email')
    df_emails_test = pd.DataFrame(emails,columns=['Emails'])
    df_emails = df_emails_test[df_emails_test['Emails'] != '']
    df_emails = df_emails['Emails'].to_list()
    st.subheader('Your Emails ')
    st.dataframe(df_emails,hide_index=True,width=1500)

    if st.button("Get Important Mails First"):
        lst = return_sorted(df_emails)
        sorted_emails = sorted(lst, key=lambda x: x[0],reverse=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.header('Here  is your mails Sorted According to their Importance')
        for i in (emails):
            st.write(i)
            st.markdown('<hr>', unsafe_allow_html=True)
        