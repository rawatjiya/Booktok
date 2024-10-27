"""SpiceGlass"""

### NB: GRAPHS AT THE BOTTOM ###

# Import libraries 
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


### CREATING THE DATA FRAME ###

filepath = "c:/Users/sonja/Downloads/spicy_df.csv"
df = pd.read_csv(filepath)
df = df.drop("rating", axis=1)

# Book title library 
book_title_mapping = {
    57936251: "Twisted Love",
    61767292: "Icebreaker",
    60755618: "Flawless",
    61373532: "Heartless",
    63031640: "Powerless",
    108518990: "Reckless", 
    174753697: "Hopeless",
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
    57693481: "Good Girl Complex",
    101145836: "Wildfire",
    199798652: "Daydream", 
    196864520: "Butcher & Blackbird",
    198716261: "Not in Love", 
    204640539: "My Vampire Plus-One", 
    139391940: "The Pumpkin Spice Café",
    209568782: "Wild Eyes",
    209773181: "Fragile Sanctuary",
    50659468: "A Court of Mist and Fury",
    50659472: "A Court of Wings and Ruin", 
    50659471: "A Court of Frost and Starlight", 
    53138095: "A ​Court of Silver Flames", 
    56554626: "The Ex Hex",
    62926938: "The Seven Year Slip", 
    202936227: "Shattered Dreams", 
    217533165: "From Here to Eternity",
    215804830: "Mad Love", 
    215639034: "The Wingman",
    174713026: "Behind the Net", 
    184591636: "The Fake Out",
    195532965: "Swift and Saddled", 
    206673322: "Lost and Lassoed",
    213650748: "Wild and Wrangled",
    209741142: "Anathema",
    204593914: "Phantasma", 
    60714999: "The Serpent and the Wings of Night", 
    58340706: "One Dark Window", 
    127305713: "Heartless Hunter", 
    202507554: "When the Moon Hatched", 
    203608430: "Lucy Undying", 
    26032825: "The Cruel Prince", 
    26032887: "The Wicked King", 
    26032912: "The Queen of Nothing", 
    61431922: "Fourth Wing",
    90202302: "Iron Flame", 
    62335396: "The Ashes & the Star-Cursed King",
    63910262: "Two Twisted Crowns", 
    60784546: "Divine Rivals", 
    203578805: "Whenever You're Ready", 
    54756850: "Delilah Green Doesn't Care",
    51179990: "Written in the Stars", 
    58800142: "Astrid Parker Doesn't Fail", 
    63826960: "Those Who Wait",
    195511768: "Cover Story", 
    62039417: "Wolfsong", 
    62039416: "Ravensong", 
    176408159: "Heartsong", 
    62039433: "Brothersong", 
    199440249: "The Pairing", 
    41150487: "Red, White & Royal Blue", 
    13623848: "The Song of Achilles", 
    32620332: "The Seven Husbands of Evelyn Hugo",
    36199084: "The Kiss Quotient", 
    61326735: "Love, Theoretically", 
    199070596: "Love Redesigned", 
    75491526: "King of Sloth", 
    217206738: "King of Wrath", 
    62994279: "King of Pride", 
    124943221: "King of Greed", 
    62022434: "Things We Hide from the Light", 
    116536542: "Things We Left Behind",
    39083635: "The Sweetest Oblivion",
    40613322: "The Maddest Obsession",
    56250307: "The Darkest Temptation"

}

# Adding a new column for book titles based on the mapping
df['book_title'] = df['book_id'].map(book_title_mapping)

##########################################


### DATA PREPROCESSING ###

# Removing punctuation
df['reviews'] = df['reviews'].str.replace(r'[^\w\s🌶️🫦🔥]+', '', regex=True)

# Removing stopwords
stopwords_list = requests.get(
    "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"
).content
stopwords = set(stopwords_list.decode().splitlines()) 

# Additional stopwords
additional_stopwords = {'book', 'books', 'dont', 'didnt', 'doesnt', 'read', 'Ive'}
stopwords.update(additional_stopwords)

# Cleaning reviews by filtering out stopwords
df['reviews'] = df['reviews'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in stopwords]))

# Stemming (not used in the final app)
#ps = PorterStemmer() # Takes a while
#df["reviews"] = df["reviews"].apply(lambda x: ' '.join([ps.stem(word) for word in str(x).split()]))

##########################################

### LABELLING AND SPLITTING ###

# List of explicit words 
explicit_words = ['🌶️', '🫦', '🔥', 'daddy', 'sex', 'bdsm', 'horny', 'dick', 'explicit', 'fuck', 'chemistry', 'explicit', 'steamy', 'spicy', 'spice', 'hot', 'erotic', 'smut', 'smutty', 'kinky', 'kink']

def label_reviews(reviews, keywords):
    """ Assigning a label to each review"""
    labels = []
    for review in reviews:
        # Check if any keyword is present in the review
        if any(keyword in review for keyword in keywords):
            labels.append("Explicit")
        else:
            labels.append("Not Explicit")
    return labels

df['label'] = label_reviews(df['reviews'], explicit_words)
X = df['reviews']  # Features (text data)
y = df['label']    # Labels (target variable)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


### BEST MODEL SEARCH ###

#from tpot import TPOTClassifier

# Initialize the TPOT classifier with sparse configuration
#tpot = TPOTClassifier(verbosity=2, random_state=42, generations=5, population_size=50, config_dict='TPOT sparse')

# Fit TPOT on the training data
#tpot.fit(X_train_tfidf, y_train)


###################################

### MODEL TRAINING ###

best_model = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=2, min_samples_split=12, n_estimators=100)
best_model.fit(X_train_tfidf, y_train)
results = best_model.predict(X_test_tfidf)

###################################

### EXTRACTING FEATURES ###

feature_importances = best_model.feature_importances_

# Sorting the scores and get the corresponding feature names
indices = np.argsort(feature_importances)[::-1]
feature_names = vectorizer.get_feature_names_out()

# Getting the top 10 important features 
important_features = [(feature_names[i], feature_importances[i]) for i in indices[:10]]
important_features_dict = {feature_names[i]: feature_importances[i] for i in indices[:10]}

#################################

### CALCULATING SPICINESS RATING ###

# List of words that correlate negatively with spice 
tame_words = ["children", "kid", "wholesome", "cozy"]

# Function to calculate spiciness score for each review
def calculate_spiciness(review, weights):
    # Initialize score
    score = 0.0
    
    # Counting occurrences of each feature in the review
    for word, weight in weights.items():
        count = review.count(word)  
        if count > 0:
            score += count * weight
        
    if any(tame_word in review for tame_word in tame_words):
        return 1.0  # Default score for children's or wholesome content
    
    # Setting maximum threshold for spiciness rating
    max_spiciness_score = 10.0  # Adjust as needed
    if score >= max_spiciness_score:
        return 5.0  # Return max rating if score exceeds threshold
    
    # Normalizing score to a rating scale (1-5)
    normalized_score = np.clip(score / max_spiciness_score * 5, 1, 5)  # Ensure score is between 1 and 5
    
    return round(normalized_score, 1)

# Creating weights from important features 
weights = {word: weight * 115 for word, weight in important_features} 

# Applying the function to each review and storing the result in a new column
df['spiciness_score'] = df['reviews'].apply(lambda review: calculate_spiciness(review, weights))

# Calculating the average spiciness score per book
average_spiciness_per_book = df.groupby('book_title')['spiciness_score'].mean().reset_index()

# Sorting by spiciness score for better readability
average_spiciness_per_book = average_spiciness_per_book.sort_values(by='spiciness_score', ascending=False)

# Mapping spiciness score to chili pepper emojis
def spiciness_to_emoji(score):

    rounded_score = int(np.round(score))
    emoji_map = {
        1: ('🫑', 'Zero spice'),
        2: ('🌶️🌶️', 'A little spicy'),
        3: ('🌶️🌶️🌶️', 'Pretty spicy'),
        4: ('🌶️🌶️🌶️🌶️', 'Steamy!'),
        5: ('🌶️🌶️🌶️🌶️🌶️', 'Scorching!!')
    }

    return emoji_map.get(rounded_score, ('🫑', 'Zero spice'))  # Default to 1 chili if no match

average_spiciness_per_book['emoji'], average_spiciness_per_book['description'] = zip(
    *average_spiciness_per_book['spiciness_score'].apply(spiciness_to_emoji)
)

### RESULT ANALYSIS ###


# Print out the spiciness ranking for each book in a formatted way
#for index, row in average_spiciness_per_book.iterrows():
#    print(f"{row['book_title']} is: {row['emoji']} {row['description']}")


#print("Top 10 important features and their weights:")
#for feature, score in important_features:
#    print(f"{feature}: {score:.4f}")

####################################

### STREAMLIT ### 
image_path = "C:/Users/sonja/Downloads/SpiceGlass.png"

st.image(image_path, use_column_width=True)
# Streamlit app title
#st.title("SpiceGlass")

book_title = st.text_input("Enter book title:")

if book_title:
    # Searching for the book in the DataFrame
    result = average_spiciness_per_book[
        average_spiciness_per_book['book_title'].str.contains(book_title, case=False)
    ]

    if not result.empty:
        book_info = result.iloc[0]
        st.write(f"{book_info['book_title']} is: {book_info['emoji']} {book_info['description']}")
    else:
        book_title = book_title.title()
        st.write(f"Sorry, '{book_title}' hasn't yet been added to our database.")

#######################################

### GRAPHS ### 

### ACCURACY & CLASSIFICATION REPORT ### 

#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#report = classification_report(y_test, results, output_dict=True)
#report_df = pd.DataFrame(report).transpose()

#print(f"Performance of {best_model}")
#accuracy_rf = accuracy_score(y_test, results)
#print(f'Accuracy: {accuracy_rf:.2f}')
#report_df

###################################

### CONFUSION MATRIX ###

#%matplotlib inline
#cm_rf = confusion_matrix(y_test, results)
#plt.figure(figsize=(8, 6))
#sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Explicit', 'Explicit'], yticklabels=['Not Explicit', 'Explicit'])
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#plt.title('Confusion Matrix for best_model')
#plt.show()

####################################

### TOP 10 FEATURES ###

#plt.figure(figsize=(10, 6))
#plt.barh(range(len(important_features)), [importance for _, importance in important_features], align='center')
#plt.yticks(range(len(important_features)), [feature for feature, _ in important_features])
#plt.xlabel('Importance Score')
#plt.title('Top 10 Important Features for Explicit Review Classification')
#plt.show()

###################################

###### WORDCLOUDS #######

#from wordcloud import WordCloud

### EXPLICIT REVIEWS ###

#explicit_reviews = df[df['label'] == 'Explicit']['reviews']

# Combining all explicit reviews into a single string
#explicit_text = ' '.join(explicit_reviews)

#wordcloud = WordCloud(width=800, height=400, background_color='white', colormap="Reds").generate(explicit_text)

#plt.figure(figsize=(10, 6))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')  # Turn off the axis
#plt.title('Word Cloud of Explicit Reviews', fontsize=16)  # Add a title
#plt.show()

####################################


### INEXPLICIT REVIEWS ###

#inexplicit_reviews = df[df['label'] == 'Not Explicit']['reviews']
#explicit_text = ' '.join(inexplicit_reviews)

#wordcloud = WordCloud(width=800, height=400, background_color='white').generate(inexplicit_text)
#plt.figure(figsize=(10, 6))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')  # Turn off the axis
#plt.title('Word Cloud of Inexplicit Reviews', fontsize=16)  # Add a title
#plt.show()

###################################


### EXPLICIT REVIEWS PER BOOK TITLE ###

#import matplotlib.pyplot as plt
#import seaborn as sns

# Function to label reviews as 'Explicit' or 'Not Explicit' based on keywords
#def label_reviews(reviews, keywords):
#    labels = []
#    for review in reviews:
#        if any(keyword in review.lower() for keyword in keywords):
#            labels.append("Explicit")
#        else:
#            labels.append("Not Explicit")
#    return labels

    
#df['label'] = label_reviews(df['reviews'], explicit_words)

# Filtering explicit reviews and count them by book title
#explicit_counts = df[df['label'] == 'Explicit'].groupby('book_title').size()

# Converting to DataFrame and resetting the index for plotting
#explicit_counts = explicit_counts.reset_index(name='Explicit_Count')

# Sorting by 'Explicit_Count' for top 10 books
#top_explicit_counts = explicit_counts.sort_values(by='Explicit_Count', ascending=False).head(10)

#plt.figure(figsize=(10, 6))
#sns.barplot(x='book_title', y='Explicit_Count', data=explicit_counts, palette='viridis')

#plt.xlabel('')
#plt.ylabel('Number of Explicit Reviews')
#plt.title('Explicit Reviews per Book Title')
#plt.xticks([])  # X-axis labels were removed because 
#plt.show()

##################################