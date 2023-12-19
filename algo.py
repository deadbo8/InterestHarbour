# Library Imports
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import QuantileTransformer
import streamlit as st
import _pickle as pickle
from random import sample
from PIL import Image
from scipy.stats import halfnorm
import math

st.set_page_config(
    page_title="InterestHarbour",
    page_icon="🤖",
    layout="wide",
)

# Loading the Profiles
with open("./ProfileData_pickleFiles/refined_profiles.pkl",'rb') as fp:
    df = pickle.load(fp)

with open("refined_cluster.pkl", 'rb') as fp:
    cluster_df = pickle.load(fp)

with open("vectorized_refined.pkl", 'rb') as fp:
    vect_df = pickle.load(fp)

# Loading the Classification Model
model = load("final_model.joblib")

## Helper Functions

def string_convert(x):
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x

def vectorization(df, columns, input_df):
    column_name = columns[0]

    if column_name not in ['Bios', 'Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Programming', 'Traveller']:
        return df, input_df

    if column_name in ['Religion', 'Politics']:
        df[column_name.lower()] = df[column_name].cat.codes
        d = dict(enumerate(df[column_name].cat.categories))
        d = {v: k for k, v in d.items()}
        input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]
        input_df = input_df.drop(column_name, axis=1)
        df = df.drop(column_name, axis = 1)
        return vectorization(df, df.columns, input_df)

    else:
        if column_name in df.columns and column_name in input_df.columns:
            vectorizer = CountVectorizer()
            x = vectorizer.fit_transform(df[column_name].values.astype('U'))
            y = vectorizer.transform(input_df[column_name].values.astype('U'))
            df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
            y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names_out(), index=input_df.index)
            new_df = pd.concat([df, df_wrds], axis=1)
            y_df = pd.concat([input_df, y_wrds], axis=1)
            if column_name in new_df.columns:
                new_df = new_df.drop(column_name, axis=1)
            if column_name in y_df.columns:
                y_df = y_df.drop(column_name, axis=1)
            return vectorization(new_df, new_df.columns, y_df)
        else:
            return vectorization(df, columns[1:], input_df)


def scaling(df, input_df):
    scaler = QuantileTransformer()
    scaler.fit(df)
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
    return input_vect


def top_ten(cluster, vect_df, input_vect):
    des_cluster = vect_df[vect_df['Cluster Labels']==cluster[0]].drop('Cluster Labels', axis=1)
    des_cluster = pd.concat([des_cluster, input_vect], axis=1, sort=False)
    user_n = input_vect.index[0]
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])
    top_10_sim = corr.sort_values(ascending=False)[1:11]
    top_10 = df.loc[top_10_sim.index]
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]]
    return top_10.astype('object')


def example_bios():
    st.write("-"*100)
    st.text("Some example Bios:\n(Try to follow the same format)")
    for i in sample(list(df.index), 3):
        st.text(df['Bios'].loc[i])
    st.write("-"*100)


# Creating a List for each Category
p = {}
tv = ['Comedy', 'Drama', 'Action/Adventure', 'Suspense/Thriller', 'Documentaries', 'Crime/Mystery', 'News', 'SciFi', 'History']
p['TV'] = [0.25, 0.21, 0.17, 0.16, 0.09, 0.08, 0.03, 0.02, 0.01]
movies = ['Adventure', 'Action', 'Drama', 'Comedy', 'Thriller', 'Horror', 'RomCom', 'Musical', 'Documentary']
p['Movies'] = [0.26, 0.21, 0.16, 0.14, 0.09, 0.06, 0.04, 0.01, 0.03]
religion = ['Catholic', 'Christian', 'Jewish', 'Muslim', 'Hindu', 'Buddhist', 'Spiritual', 'Other', 'Agnostic', 'Atheist']
p['Religion'] = [0.07, 0.13, 0.01, 0.19, 0.24, 0.05, 0.10, 0.09, 0.07, 0.05]
music = ['Rock', 'HipHop', 'Romantic', 'Pop', 'Country', 'EDM', 'Jazz', 'Classical']
p['Music'] = [0.25, 0.19, 0.16, 0.14, 0.10, 0.06, 0.04, 0.03, 0.02, 0.01,]
sports = ['Cricket', 'Chess', 'Badminton' 'Football', 'Baseball', 'Basketball', 'Hockey', 'Soccer', 'Other']
p['Sports'] = [0.29, 0.24, 0.23, 0.13, 0.04, 0.03, 0.02, 0.02]
politics = ['Liberal', 'Progressive', 'Centrist', 'Moderate', 'Conservative']
p['Politics'] = [0.26, 0.11, 0.11, 0.15, 0.37]
social = ['Facebook', 'Youtube', 'Twitter', 'Reddit', 'Instagram', 'Pinterest', 'LinkedIn', 'SnapChat', 'TikTok']
p['Social Media'] = [0.36, 0.27, 0.11, 0.09, 0.05, 0.03, 0.03, 0.03, 0.03]
programming = ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Swift', 'Go']
p['programming'] = [0.23, 0.20, 0.18, 0.15, 0.12, 0.10, 0.02]
travelling = ['Treking', 'Adventure', 'Long Trips', 'Short journeys']
p['travelling'] = [0.35, 0.33, 0.21, 0.11]
age=None
categories = [movies, religion, music, politics, social, sports,programming, travelling, age]
names = ['Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Programming', 'Traveller', 'Age']
combined = dict(zip(names, categories))


# Interactive Section
col1, col2 = st.columns([1, 2])

# In the first column, display the image and text horizontally
col1.image("1.jpeg", width=150)
col1.text("")

# In the second column, display the headers with improved styling
col2.header("InterestHarBour")
col2.subheader("Finding connections with Artificial Intelligence")


# ...

st.write("Use machine learning to find people with similar interests.")

new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])

new_profile['Bios'] = st.text_input("Enter a Bio for yourself:")

example_bios()

random_vals = st.checkbox("Check here if you would like random values for yourself instead")

if random_vals:
    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = np.random.choice(combined[i], 1, p=p[i])

        elif i == 'Age':
            new_profile[i] = halfnorm.rvs(loc=18,scale=8, size=1).astype(int)

        else:
            new_profile.at[new_profile.index[0], i] = list(np.random.choice(combined[i], size=3, p=p[i], replace=False))


            new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))

else:
    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = st.selectbox(f"Enter your choice for {i}:", combined[i])

        elif i == 'Age':
            new_profile[i] = st.slider("What is your age?", 18, 50)

        else:
            options = st.multiselect(f"What is your preferred choice for {i}? (Pick up to 3)", combined[i])
            new_profile.at[new_profile.index[0], i] = options
            new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))

for col in df.columns:
    df[col] = df[col].apply(string_convert)
    new_profile[col] = new_profile[col].apply(string_convert)

st.write("-"*1000)
st.write("Your profile:")
st.table(new_profile)

button = st.button("Click to find your Top 10!")

if button:    
    with st.spinner('Finding your Top 10 Matches...'):
        df_v, input_df = vectorization(df, df.columns, new_profile)
        new_df = scaling(df_v, input_df)
        cluster = model.predict(new_df)
        top_10_df = top_ten(cluster, vect_df, new_df)
        st.success("Found your Top 10 Most Similar Profiles!")    
        st.balloons()

        st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" 
            integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
        """,unsafe_allow_html=True)

        def card(Bios,Movies,Religion,Music,Politics,SocialMedia,Sports,Programming, Traveller, Age ,v1, v2, v3, v4,v5,v6,v7,v8,v9):
            return f'''
            <div class="card text-center" style="display: flex; align-content:flex-start ;flex-direction: column; width:18rem">
                <div class="card-body">
                    <div class="card-title" style="color: Black;">Bio: {Bios}</div>
                    <div class="card-subtitle mb-2 text-muted" style="display: {v1};color: Black;">Movies: {Movies}</div>
                    <div class="card-text" style="display: {v2};color: Black;">Religion: {Religion}</div>
                    <div class="card-text" style="display: {v3};color: Black;">Music: {Music}</div>
                    <div class="card-text" style="display: {v4};color: Black;">Politics: {Politics}</div>
                    <div class="card-text" style="display: {v5};color: Black;">Social Media: {SocialMedia}</div>
                    <div class="card-text" style="display: {v6};color: Black;">Sports: {Sports}</div>
                    <div class="card-text" style="display: {v7};color: Black;">Programming Languages: {Programming}</div>
                    <div class="card-text" style="display: {v8};color: Black;">Travelling: {Traveller}</div>
                    <div class="card-text" style="display: {v9};color: Black;">Age: {Age}</div>
                </div>
            </div>
            '''

        df_result_search = pd.DataFrame()
        with st.sidebar.form(key='search_form',clear_on_submit= False):
            regular_search_term=st.text_input("Enter Search Term")
            if st.form_submit_button("search"):
                df_result_search = df[df['name'].str.contains(regular_search_term) |
                                    df['nickname'].str.contains(regular_search_term) |
                                    df['mother_name'].str.contains(regular_search_term)]

        rec = top_10_df.shape[0]
        st.write("{} Records ".format(str(rec)))

        num_rows = max(1, math.ceil(rec/5))

        dv = 'flex'
        cards = []
        for index, row in top_10_df.iterrows():
            v1 = dv if row[1] is not None else 'none'
            v2 = dv if row[2] is not None else 'none'
            v3 = dv if row[3] is not None else 'none'
            v4 = dv if row[4] is not None else 'none'
            v5 = dv if row[4] is not None else 'none'
            v6 = dv if row[4] is not None else 'none'
            v7 = dv if row[4] is not None else 'none'
            v8 = dv if row[4] is not None else 'none'
            v9 = dv if row[4] is not None else 'none'
            cards.append([row[0], row[1], row[2], row[3], row[4],row[5],row[6],row[7],row[8],row[9],v1, v2, v3, v4,v5,v6,v7,v8,v9])

        if len(cards):
            for r in range(num_rows):
                num_cols = 10
                c = st.columns(num_cols)
                for m in range(num_cols):
                    if len(cards):
                        mycard = cards.pop(0)
                        c[m].markdown(card(*mycard), unsafe_allow_html=True)

        st.table(top_10_df)
