import streamlit as st
import pickle
import requests
import pandas as pd

# Load the pickled data
try:
    movies = pickle.load(open("movies_list.pkl", 'rb'))
    similarity = pickle.load(open("similarity.pkl", 'rb'))
    movies_list = movies['title'].values
except (FileNotFoundError, ValueError) as e:
    st.error(f"Error loading model files: {e}. Please run the `build_model.py` script first.")
    st.stop()

def fetch_poster(movie_id):
    """Fetches movie poster from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
    try:
        data = requests.get(url)
        data.raise_for_status() # Raise an exception for bad status codes
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return full_path
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching movie poster: {e}")
    return None

def recommend(movie):
    """Generates a list of recommended movies and their posters."""
    try:
        index = movies[movies['title'] == movie].index[0]
        distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
        recommend_movie = []
        recommend_poster = []
        for i in distance[1:6]:
            movies_id = movies.iloc[i[0]].id
            recommend_movie.append(movies.iloc[i[0]].title)
            recommend_poster.append(fetch_poster(movies_id))
        return recommend_movie, recommend_poster
    except IndexError:
        st.warning(f"Movie '{movie}' not found in the dataset.")
        return [], []

# Streamlit App UI
st.title("Movie Recommender System")
st.markdown("Select a movie from the dropdown to get recommendations based on content similarity.")

# Display a carousel of movie posters
st.subheader("Featured Movies")
image_urls = [
    fetch_poster(1632),
    fetch_poster(299536),
    fetch_poster(17455),
    fetch_poster(2830),
    fetch_poster(429422)
]
# Filter out any None values from failed fetches
image_urls = [url for url in image_urls if url is not None]

# This is a simplified display without a complex component
if image_urls:
    cols = st.columns(len(image_urls))
    for col, url in zip(cols, image_urls):
        with col:
            st.image(url, use_column_width=True)

# Main recommendation interface
select_value = st.selectbox("Select a movie from the dropdown", movies_list)

if st.button("Show Recommendations"):
    movie_name, movie_poster = recommend(select_value)
    if movie_name:
        st.subheader(f"Recommended Movies for '{select_value}'")
        cols = st.columns(len(movie_name))
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"**{movie_name[i]}**")
                if movie_poster[i]:
                    st.image(movie_poster[i])
                else:
                    st.image("https://placehold.co/500x750?text=Poster+Not+Found", use_column_width=True)
