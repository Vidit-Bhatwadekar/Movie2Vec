import streamlit as st
import pickle
import pandas as pd
import string
import numpy as np

st.set_page_config(
    page_title="Movie2Vec - Movie Recommendation engine",
    page_icon= "🎥"
)

st.markdown('''<style> 
ul {
    padding-left: 1.5rem;
  }
h1 {
    text-align:center;
}
h5 {
    text-align:center;
}
img {
    margin-left: 17.2rem;
}
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
header {
    visibility: hidden;
}')
</style>
''', unsafe_allow_html=True)

st.title("Movie2Vec Webapp", anchor=None)

st.write('''
We built this webapp as a part of our final deliverable for a Data Mining and Machine Learning course. 
Using a dataset of 10000 movie descriptions, we built a Movie2Vec model which embeds each movie as a 100 dimensional vector.
Using this model, we created an app that suggests movies based on a description or some key words that you input
You can use these vectors to add and subtract movies from each other to help you find new and interesting movies that you may
have never seen.
''')

df = pd.read_csv("./data/github_csv.csv", lineterminator="\n", index_col=0)
model = pickle.load(open('newest_model.pkl', 'rb'))
df["vec"] = [v for v in model.dv.vectors]



def find_similar_movies(description):
    '''
    Using description find similar movies
    '''
    description.lower()
    description.replace(r'[^\w\s]','')
    words = description.split()
    vec = model.infer_vector(words)
    closest = model.dv.most_similar(positive=vec)
    closest_titles = [(df.loc[n, "original_title"]) for n in [index[0] for index in closest]]
    return closest_titles

# dune = "Paul Atreides, a brilliant and gifted young man born into a great destiny beyond his understanding, must travel to the most dangerous planet in the universe to ensure the future of his family and his people. As malevolent forces explode into conflict over the planet's exclusive supply of the most precious resource in existence-a commodity capable of unlocking humanity's greatest potential-only those who can conquer their fear will survive."
# dune_similar = find_similar_movies(dune)

with st.container():
    st.subheader("Movie Finder")
    st.write("Input key-words or a description about the type of movie you would like to watch and we will suggest movies below")


    description = st.text_area("input a description")
    if description:
        similar_movies = find_similar_movies(description)
        for movie in similar_movies:
            st.text(movie)


def find_similar_movies_addition(positive_list, negative_list):
    add_vecs = []
    sub_vecs = []
    
    for movie in positive_list:
        vec = df[df["original_title"]==movie]["vec"].iloc[0]
        add_vecs.append(vec)
    for movie in negative_list:
        vec = df[df["original_title"]==movie]["vec"].iloc[0]
        sub_vecs.append(vec)
    
    movie_suggested_tuples = model.dv.most_similar(positive=add_vecs, negative=sub_vecs)
    suggested_movies = [df.loc[movie_tuple[0],"original_title"] for movie_tuple in movie_suggested_tuples]
    return suggested_movies



with st.container():
    st.subheader("Movie Arithmatic")
    st.write("With this tool you can add or subtract movies to find another movie")

    add_list = st.multiselect("Select the movies you'd like to add", df["original_title"].unique())
    add_list = list(add_list)
    sub_list = st.multiselect("Select the movies you'd like to subtract", df["original_title"].unique())
    sub_list = list(sub_list)

    if add_list or sub_list:
        similar_movies_addition = find_similar_movies_addition(add_list, sub_list)
        temp_list = set(similar_movies_addition)
        similar_movies_addition = list(dict.fromkeys(similar_movies_addition))
        for movie in similar_movies_addition:
            st.write(movie)


