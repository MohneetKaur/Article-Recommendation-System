from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the shared_articles_df DataFrame from pickle file
shared_articles_df = pickle.load(open('article_df.pkl', 'rb'))

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Create TF-IDF matrix for the article text
tfidf_matrix = tfidf_vectorizer.fit_transform(shared_articles_df['text'])

# Load the cosine similarity matrix from pickle file
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

# Function to get top N similar articles based on text similarity
def get_similar_articles(title, cosine_sim_matrix, N):
    # Transform the input text into a TF-IDF vector
    input_vector = tfidf_vectorizer.transform([title])

    # Calculate the similarity scores between the input vector and all articles
    sim_scores = cosine_similarity(input_vector, tfidf_matrix)

    # Sort the articles based on similarity scores
    sim_scores = sim_scores.flatten()
    sorted_indices = sim_scores.argsort()[::-1]

    # Get the top N similar articles
    top_articles = sorted_indices[1:N+1]

    # Extract the desired information for the recommended articles
    recommended_articles = shared_articles_df.iloc[top_articles]
    recommended_articles = recommended_articles[['title', 'url']]

    # Return the recommended articles
    return recommended_articles

# Rest of the Flask app code...


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the input title from the form
    title = request.form['title']

    # Number of recommendations to generate
    num_recommendations = 5

    # Get the recommended articles
    recommended_articles = get_similar_articles(title, cosine_sim, num_recommendations)

    # Convert the recommended articles to a list of dictionaries
    recommendations = recommended_articles.to_dict('records')

    return render_template('recommend.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
