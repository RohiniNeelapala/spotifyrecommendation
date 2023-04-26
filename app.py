from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the Spotify dataset
df = pd.read_csv('spotify.csv')

# Define the input fields for the web form
@app.route('/')
def home():
    return render_template('Index.html')

# Handle the form submission
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get input from user
    song_name = request.form['song_name']
    recommendations_count = int(request.form['recommendations_count'])
    
    # Filter the dataset to get the features of the input song
    input_song = df[df['song'].str.lower() == song_name.lower()]
    
    # Check if input_song exists in the dataset
    if input_song.empty:
        # Find songs with similar names and pronunciations
        similar_songs = df[df['song'].str.contains(song_name, case=False) |
                           df['song'].str.lower().str.replace('[^a-zA-Z0-9]', '').str.contains(song_name.lower().replace(' ', '').replace('-', ''), regex=False)]
        
        # Get the recommendations from the similar songs
        recommendations = similar_songs[['artists', 'song']].head(recommendations_count).drop_duplicates(subset='artists')
        
        return render_template('similar_songs.html', song_name=song_name, recommendations=recommendations)
    else:
        # Get the features of the input song
        input_features = input_song.iloc[0][['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                                  'loudness', 'speechiness', 'tempo', 'valence']]
        
        # Fit a k-nearest neighbors model on the dataset
        nbrs = NearestNeighbors(n_neighbors=recommendations_count, metric='euclidean').fit(df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                                  'loudness', 'speechiness', 'tempo', 'valence']])
        
        # Find the indices of the closest songs to the input song
        distances, indices = nbrs.kneighbors(input_features.values.reshape(1, -1))
        
        # Get the recommendations from the indices
        recommendations = df.iloc[indices.flatten()][['artists', 'song']]
        
        # Remove duplicates and convert to list
        recommendations = recommendations.drop_duplicates(subset='artists')
        
        return render_template('recommendations.html', song_name=song_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
