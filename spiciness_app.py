import numpy as np
import pandas as pd
import requests
import streamlit as st
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive

filepath = "spicy_df.csv"
df = pd.read_csv(filepath)

book_name_mapping = {
    57936251: "Twisted Love",
    61767292: "Icebreaker",
    60755618: "Flawless",
    60060431: "Things We Never Got Over",
    195820807: "Just for the Summer", 
    60909831: "Mile High", 
    199368721: "Wild Love", 
    194802722: "Funny Story", 
    215950185: "Natural Selection", 
    50659467: "A Court of Thorns and Roses", 
    199798652: "Daydream", 
    17788401: "Ugly Love", 
    60683957: "Check & Mate",
    181344829: "Bride",
    211835409: "Redeeming",
    40944965: "Binding 13", 
    48763676: "Alpha's Prey",
    46261182: "The Awakening", 
    123008168: "My Dark Romeo",
    208415817: "God of Malice",
    57426932: "Gothikana",
    200124498: "Pucking Sweet",
    208457093: "Fall with Me", 
    201145400: "Love Unwritten", 
    216657877: "Passenger Princess",
    75513900: "Powerless",
    61242426: "Legends & Lattes",
    6294: "Howl’s Moving Castle", 
    157993: "The Little Prince", 
    18144590: "The Alchemist",
    10818853: "Fifty Shades of Grey",
    212127083: "Quicksilver",
    57693481: "Good Girl Complex"
}

# Add a new column for book names based on the mapping
df['book_name'] = df['book_id'].map(book_name_mapping)

# Clean reviews
df['reviews'] = df['reviews'].str.replace(r'[^\w\s]+', '', regex=True)
stopwords_list = requests.get(
    "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"
).content
stopwords = set(stopwords_list.decode().splitlines()) 
additional_stopwords = {'book', 'books', 'dont', 'didnt', 'doesnt', 'read'}
stopwords.update(additional_stopwords)

df["reviews"] = df['reviews'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stopwords)]))

explicit_words = ['sex', 'bdsm', 'horny', 'chemistry', 'explicit', 'steamy', 'spicy', 'spice', 'hot', 'erotic', 'smut', 'smutty', 'kinky', 'kink']

def label_reviews(reviews, keywords):
    labels = []
    for review in reviews:
        # Check if any keyword is present in the review
        if any(keyword in review for keyword in keywords):
            labels.append("Explicit")
        else:
            labels.append("Not Explicit")
    return labels

# Generate labels for the existing reviews in the DataFrame
df['label'] = label_reviews(df['reviews'], explicit_words)

X = df['reviews']  # Features (text data)
y = df['label']    # Labels (target variable)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=86),
    SelectPercentile(score_func=f_classif, percentile=1),
    BernoulliNB(alpha=0.001, fit_prior=True)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(X_train_tfidf, y_train)
results = exported_pipeline.predict(X_test_tfidf)

select_percentile = exported_pipeline.steps[0][1]

# Get the scores for the features from the first SelectPercentile
feature_scores = select_percentile.scores_

# Sort the scores and get the corresponding feature names
indices_1 = np.argsort(feature_scores)[::-1]
feature_names = vectorizer.get_feature_names_out()

# Get the top 10 important features based on the SelectPercentile scores
important_features = [(feature_names[i], feature_scores[i]) for i in indices_1[:10]]
important_features_dict = {feature_names[i]: feature_scores[i] for i in indices_1[:10]}

# Function to calculate spiciness score for each review
def calculate_spiciness(review, weights):
    # Initialize score
    score = 0.0
    
    # Count occurrences of each feature in the review
    for word, weight in weights.items():
        count = review.count(word)  
        if count > 0:
            score += count * weight
            
    # Adjust conditions for children's books
    if 'children' in review or 'kid' in review:
        return 1.0  # Default score for children's content
    
    # Set maximum threshold for spiciness rating
    max_spiciness_score = 10.0  # Adjust as needed
    if score >= max_spiciness_score:
        return 5.0  # Return max rating if score exceeds threshold
    
    # Normalize score to a rating scale (1-5)
    normalized_score = np.clip(score / max_spiciness_score * 5, 1, 5)  # Ensure score is between 1 and 5
    
    return round(normalized_score, 1)

# Assuming 'important_features' holds the feature words and their weights
words_to_drop = ['love'] 
weights = {word: weight for word, weight in important_features if word not in words_to_drop} # Use your actual weights from the TF-IDF

# Apply the function to each review and store the result in a new column
df['spiciness_score'] = df['reviews'].apply(lambda review: calculate_spiciness(review, weights))

# Calculate the average spiciness score per book
average_spiciness_per_book = df.groupby('book_name')['spiciness_score'].mean().reset_index()

# Sort by spiciness score for better readability
average_spiciness_per_book = average_spiciness_per_book.sort_values(by='spiciness_score', ascending=False)

# Mapping spiciness score to chili pepper emojis
def spiciness_to_emoji(score):
    # Round the score to the nearest integer
    rounded_score = int(np.round(score))
    
    # Map the rounded score to corresponding emoji representation
    emoji_map = {
        1: '🫑',          # 1 -> One bell pepper (non-spicy)
        2: '🌶️🌶️',      # 2 -> Two chili peppers (mild)
        3: '🌶️🌶️🌶️',   # 3 -> Three chili peppers (medium)
        4: '🌶️🌶️🌶️🌶️', # 4 -> Four chili peppers (hot)
        5: '🌶️🌶️🌶️🌶️🌶️' # 5 -> Five chili peppers (very hot)
    }
    
    # Return the emoji string corresponding to the rounded score
    return emoji_map.get(rounded_score, '🫑')  # Default to 1 chili if no match

# Apply the emoji function to the average spiciness score per book
average_spiciness_per_book['spiciness_rating'] = average_spiciness_per_book['spiciness_score'].apply(spiciness_to_emoji)

# Print out the spiciness ranking for each book
print(average_spiciness_per_book[['book_name', 'spiciness_score', 'spiciness_rating']])

print("Top 10 important features and their weights:")
for feature, score in important_features:
    print(f"{feature}: {score:.4f}")


# Streamlit app title
st.title("Book Spiciness Ratings")

# Display the DataFrame in the app
#st.dataframe(average_spiciness_per_book)

# Add a text input for user to search for a book
book_name = st.text_input("Enter the name of the book you'd like to check:")

if book_name:
    # Search for the book in the DataFrame
    result = average_spiciness_per_book[
        average_spiciness_per_book['book_name'].str.contains(book_name, case=False)
    ]

    if not result.empty:
        book_info = result.iloc[0]
        st.write(f"The spiciness rating for '{book_info['book_name']}' is: {book_info['spiciness_rating']}")
    else:
        st.write(f"Sorry, the book '{book_name}' was not found in the dataset.")