import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

# Load the dataset
movies = pd.read_csv('dataset.csv')

# Drop duplicate rows
movies.drop_duplicates(subset=['title'], keep='first', inplace=True)

# Combine relevant features into a single string
movies['combined_features'] = movies['genre'].fillna('') + ' ' + movies['overview'].fillna('')

# Create a TF-IDF Vectorizer
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 2), stop_words='english')

# Fit and transform the data
tfv_matrix = tfv.fit_transform(movies['combined_features'])

# Calculate the cosine similarity matrix
cosine_sim = linear_kernel(tfv_matrix, tfv_matrix)

# Create a list of movie titles
movies_list = movies[['id', 'title']].reset_index(drop=True)

# Save the model and movie list using pickle
pickle.dump(movies_list, open('movies_list.pkl', 'wb'))
pickle.dump(cosine_sim, open('similarity.pkl', 'wb'))

print("Successfully generated movies_list.pkl and similarity.pkl")