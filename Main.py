import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
metadata.head(3)  # tell pandas how much is a header

# Calculate the mean of the vote_average column
C = metadata['vote_average'].mean()

print(f"Average: {C}")
# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print(f"Minimum votes: {m}")
# filter qualified movies to a new dataframe, q_movies
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print(f"q_movies shape: {q_movies.shape}")
print(f"metadata shape: {metadata.shape}")


# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)

# print the top 20 movies
# print("Top 20 Movies")
# print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))
#
# print(metadata['overview'].head(5))

# Define a TF-IDF Vectoriser Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of tfidf_matrix
print(f"TF-IDF shape: {tfidf_matrix.shape}")
print(tfidf.get_feature_names_out()[5000:5010])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(f"Cosine_sim Shape: {cosine_sim.shape}")
print(cosine_sim[1])