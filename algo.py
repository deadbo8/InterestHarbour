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
    """
    First converts the lists in the DF into strings
    """
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x
 
    
def vectorization(df, columns, input_df):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """

    column_name = columns[0]
        
    # Checking if the column name has been removed already
    if column_name not in ['Bios', 'Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Programming', 'Traveller']:
                
        return df, input_df
    
    # Encoding columns with respective values
    if column_name in ['Religion', 'Politics']:
        
        # Getting labels for the original df
        df[column_name.lower()] = df[column_name].cat.codes
        
        # Dictionary for the codes
        d = dict(enumerate(df[column_name].cat.categories))
        
        d = {v: k for k, v in d.items()}
                
        # Getting labels for the input_df
        input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]
                
        # Dropping the column names
        input_df = input_df.drop(column_name, axis=1)
        
        df = df.drop(column_name, axis = 1)
        
        return vectorization(df, df.columns, input_df)
    
    # Vectorizing the other columns
    # Vectorizing the other columns
    # Vectorizing the other columns
    else:
        # Check if the column is present in both training and prediction data
        if column_name in df.columns and column_name in input_df.columns:
            # Instantiating the Vectorizer
            vectorizer = CountVectorizer()

            # Fitting the vectorizer to the columns
            x = vectorizer.fit_transform(df[column_name].values.astype('U'))

            y = vectorizer.transform(input_df[column_name].values.astype('U'))

            # Creating a new DF that contains the vectorized words
            df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

            y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names_out(), index=input_df.index)

            # Concatenating the words DF with the original DF
            new_df = pd.concat([df, df_wrds], axis=1)

            y_df = pd.concat([input_df, y_wrds], axis=1)

            # Dropping the column because it is no longer needed in place of vectorization
            if column_name in new_df.columns:
                new_df = new_df.drop(column_name, axis=1)

            if column_name in y_df.columns:
                y_df = y_df.drop(column_name, axis=1)

            return vectorization(new_df, new_df.columns, y_df)
        else:
            # If the column is not present in both training and prediction data, skip it
            return vectorization(df, columns[1:], input_df)

 

    
def scaling(df, input_df):
    """
    Scales the new data with the scaler fitted from the previous data
    """
    scaler = QuantileTransformer()
    
    scaler.fit(df)
    
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
        
    return input_vect
    


def top_ten(cluster, vect_df, input_vect):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    # Filtering out the clustered DF
    des_cluster = vect_df[vect_df['Cluster Labels']==cluster[0]].drop('Cluster Labels', axis=1)
    
    # Appending the new profile data
    des_cluster = pd.concat([des_cluster, input_vect], axis=1, sort=False)

        
    # Finding the Top 10 similar or correlated users to the new user
    user_n = input_vect.index[0]
    
    # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])

    # Creating a DF with the Top 10 most similar profiles
    top_10_sim = corr.sort_values(ascending=False)[1:11]
        
    # The Top Profiles
    top_10 = df.loc[top_10_sim.index]
        
    # Converting the floats to ints
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]]
    
    return top_10.astype('object')


def example_bios():
    """
    Creates a list of random example bios from the original dataset
    """
    # Example Bios for the user
    st.write("-"*100)
    st.text("Some example Bios:\n(Try to follow the same format)")
    for i in sample(list(df.index), 3):
        st.text(df['Bios'].loc[i])
    st.write("-"*100)

## Creating a List for each Category

# Probability dictionary
p = {}

# TV Genres
tv = ['Comedy',
      'Drama',
      'Action/Adventure',
      'Suspense/Thriller',
      'Documentaries',
      'Crime/Mystery',
      'News',
      'SciFi',
      'History']

p['TV'] = [0.25,
           0.21,
           0.17,
           0.16,
           0.09,
           0.08,
           0.03,
           0.02,
           0.01]

# Movie Genres
movies = ['Adventure',
          'Action',
          'Drama',
          'Comedy',
          'Thriller',
          'Horror',
          'RomCom',
          'Musical',
          'Documentary']

p['Movies'] = [0.26,
               0.21,
               0.16,
               0.14,
               0.09,
               0.06,
               0.04,
               0.01, 
               0.03]

# Religions (could potentially create a spectrum)
religion = ['Catholic',
            'Christian',
            'Jewish',
            'Muslim',
            'Hindu',
            'Buddhist',
            'Spiritual',
            'Other',
            'Agnostic',
            'Atheist']

p['Religion'] = [0.07,
                 0.13,
                 0.01,
                 0.19,
                 0.24,
                 0.05,
                 0.10,
                 0.09,
                 0.07,
                 0.05]

# Music
music = ['Rock',
         'HipHop',
         'Romantic',
         'Pop',
         'Country',
         'EDM',
         'Jazz',
         'Classical',
         ]

p['Music'] = [0.25,
              0.19,
              0.16,
              0.14,
              0.10,
              0.06,
              0.04,
              0.03,
              0.02,
              0.01,]

# Sports
sports = [
          'Cricket',
          'Chess',
          'Badminton'
          'Football',
          'Baseball',
          'Basketball',
          'Hockey',
          'Soccer',
          'Other']

p['Sports'] = [0.29,
               0.24,
               0.23, 
               0.13,
               0.04,
               0.03,
               0.02,
               0.02]

# Politics (could also put on a spectrum)
politics = ['Liberal',
            'Progressive',
            'Centrist',
            'Moderate',
            'Conservative']

p['Politics'] = [0.26,
                 0.11,
                 0.11,
                 0.15,
                 0.37]

# Social Media
social = ['Facebook',
          'Youtube',
          'Twitter',
          'Reddit',
          'Instagram',
          'Pinterest',
          'LinkedIn',
          'SnapChat',
          'TikTok']

p['Social Media'] = [0.36,
                     0.27,
                     0.11,
                     0.09,
                     0.05,
                     0.03,
                     0.03,
                     0.03,
                     0.03]

# Programming
programming = [
          'Python',
          'Java',
          'JavaScript',
          'C++',
          'C#',
          'Swift',
          'Go'
         ]

p['programming'] = [
                    0.23,
                    0.20,
                    0.18,
                    0.15,
                    0.12,
                    0.10,
                    0.02
                   ]

#travelling
travelling = [
            'Treking',
            'Adventure',
            'Long Trips',
            'Short journeys'
             ]

p['travelling'] = [
                 0.35,
                 0.33,
                 0.21,
                 0.11
                  ]


# Age (generating random numbers based on half normal distribution)
# age = halfnorm.rvs(loc=18,scale=8, size=df.shape[0]).astype(int)
age=None

# Lists of Names and the list of the lists
categories = [movies, religion, music, politics, social, sports,programming, travelling, age]

names = ['Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Programming', 'Traveller', 'Age']

combined = dict(zip(names, categories))
    
    
## Interactive Section

# Creating the Titles and Image
# st.title("AI-MatchMaker")

# st.header("Finding a Date with Artificial Intelligence")
st.write("Using Machine Learning to Find People with similar interests")

# image = Image.open('robot_matchmaker.jpg')

# st.image(image, use_column_width=True)

# Instantiating a new DF row to classify later
new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])

# Asking for new profile data
new_profile['Bios'] = st.text_input("Enter a Bio for yourself: ")

# Printing out some example bios for the user        
example_bios()

# Checking if the user wants random bios instead
random_vals = st.checkbox("Check here if you would like random values for yourself instead")

# Entering values for the user
if random_vals:
    # Adding random values for new data
    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = np.random.choice(combined[i], 1, p=p[i])
            
        elif i == 'Age':
            new_profile[i] = halfnorm.rvs(loc=18,scale=8, size=1).astype(int)
            
        else:
            new_profile[i] = list(np.random.choice(combined[i], size=(1,3), p=p[i]))
            
            new_profile[i] = new_profile[i].apply(lambda x: list(set(x.tolist())))

else:
    # Manually inputting the data
    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = st.selectbox(f"Enter your choice for {i}:", combined[i])
            
        elif i == 'Age':
            new_profile[i] = st.slider("What is your age?", 18, 50)
            
        else:
            options = st.multiselect(f"What is your preferred choice for {i}? (Pick up to 3)", combined[i])
            
            # Assigning the list to a specific row
            new_profile.at[new_profile.index[0], i] = options
            
            new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))
            
            
# Looping through the columns and applying the string_convert() function (for vectorization purposes)
for col in df.columns:
    df[col] = df[col].apply(string_convert)
    
    new_profile[col] = new_profile[col].apply(string_convert)
            

# Displaying the User's Profile 
st.write("-"*1000)
st.write("Your profile:")
st.table(new_profile)

# Push to start the matchmaking process
button = st.button("Click to find your Top 10!")

if button:    
    with st.spinner('Finding your Top 10 Matches...'):
        # Vectorizing the New Data
        df_v, input_df = vectorization(df, df.columns, new_profile)
                
        # Scaling the New Data
        new_df = scaling(df_v, input_df)
                
        # Predicting/Classifying the new data
        cluster = model.predict(new_df)
        
        # Finding the top 10 related profiles
        top_10_df = top_ten(cluster, vect_df, new_df)
        
        # Success message   
        st.success("Found your Top 10 Most Similar Profiles!")    
        st.balloons()

        # data = {
        #     'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'name': ['peter', 'john', 'james', 'james', 'james', 'james', 'james', 'james', 'james'],
        #     'nickname': ['pet', 'jon', None, 'james', 'jem', 'jack', 'jo', 'mes', 'ja'],
        #     'mother_name': ['maria', 'linda', 'ana', 'beth', 'vivian', 'rose', 'meg', 'sharon', 'neda'],
        #     'bd': ['2000-05-15', '2006-09-12', '2004-10-25', '1999-01-08', '2009-12-14', '2001-11-20', '1999-05-12', '2009-01-28', '2009-03-14']    
        # }

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

        df = pd.DataFrame(top_10_df)
        st.write(top_10_df)

        # df_result_search = pd.DataFrame()
        # with st.sidebar.form(key='search_form',clear_on_submit= False):
        #     regular_search_term=st.text_input("Enter Search Term")
        #     if st.form_submit_button("search"):
        #         df_result_search = df[df['name'].str.contains(regular_search_term) |
        #                             df['nickname'].str.contains(regular_search_term) |
        #                             df['mother_name'].str.contains(regular_search_term)]
                                
        rec = top_10_df.shape[0]
        st.write("{} Records ".format(str(rec)))

        num_rows = max(1, math.ceil(rec/5))

        # Save cards data temporarily.
        dv = 'flex'
        cards = []
        for index, row in top_10_df.iterrows():
            print(row)
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

        # Show the card data.
        if len(cards):
            for r in range(num_rows):
                num_cols = 10
                c = st.columns(num_cols)
                for m in range(num_cols):
                    if len(cards):
                        mycard = cards.pop(0)
                        c[m].markdown(card(*mycard), unsafe_allow_html=True)
                        
        # Displaying the Top 10 similar profiles
        st.table(top_10_df)
        
        

        

    
