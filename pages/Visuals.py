### GRAPHS ### 
# Importing needed librarires
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from model import important_features, X_train, X_test, y_train, y_test, results, best_model, df, explicit_words

st.title("Data visualisation")
st.markdown("Below are graphs visualising our data in different ways.")
st.markdown("\n\n")

### ACCURACY & CLASSIFICATION REPORT ### 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.subheader("Accuracy & Classification Report")

report = classification_report(y_test, results, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print(f"Performance of {best_model}")
accuracy_rf = accuracy_score(y_test, results)
print(f'Accuracy: {accuracy_rf:.2f}')
report_df

st.caption("Shows the performance metrics for each class, along with the overall accuracy of the model.")
st.markdown("\n\n")

###################################

### CONFUSION MATRIX ###

st.subheader("Confusion Matrix")

cm_rf = confusion_matrix(y_test, results)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds', xticklabels=['Not Explicit', 'Explicit'], yticklabels=['Not Explicit', 'Explicit'], ax=ax)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Confusion Matrix for best_model')
st.pyplot(fig)

st.caption("Visualises the counts of true positive, true negative, false positive, and false negative predictions made by the model.")
st.markdown("\n\n")

####################################

### TOP 10 FEATURES ###

st.subheader("Top 10 Features")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(important_features)), [importance for _, importance in important_features], align='center', color='#d40000')
ax.set_yticks(range(len(important_features)))
ax.set_yticklabels([feature for feature, _ in important_features])
ax.set_xlabel('Importance Score')
ax.set_title('Top 10 Important Features for Explicit Review Classification')
st.pyplot(fig)

st.caption("Displays the most influential features used by the model to classify reviews as explicit or non-explicit, ranked by their importance scores.")
st.markdown("\n\n")

###################################

###### WORDCLOUDS #######

st.subheader("WordClouds")

from wordcloud import WordCloud

### EXPLICIT REVIEWS ###

explicit_reviews = df[df['label'] == 'Explicit']['reviews']

# Combining all explicit reviews into a single string
explicit_text = ' '.join(explicit_reviews)

fig, ax = plt.subplots(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap="Reds").generate(explicit_text)
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')  # Turn off the axis
ax.set_title('Word Cloud of Explicit Reviews', fontsize=16)  # Add a title
st.pyplot(fig)

st.caption("Highlights commonly occuring words within explicit reviews. Frequenlty used words are larger.")
st.markdown("\n\n")

####################################


### INEXPLICIT REVIEWS ###

inexplicit_reviews = df[df['label'] == 'Not Explicit']['reviews']
inexplicit_text = ' '.join(inexplicit_reviews)

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(inexplicit_text)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')  # Turn off the axis
ax.set_title('Word Cloud of Inexplicit Reviews', fontsize=16)  # Add a title
st.pyplot(fig)

st.caption("Highlights commonly occuring words within inexplicit reviews. Frequenlty used words are larger.")
st.markdown("\n\n")

###################################


### EXPLICIT REVIEWS PER BOOK TITLE ###

st.subheader("Explicit Reviews per Book Title")

# Function to label reviews as 'Explicit' or 'Not Explicit' based on keywords
def label_reviews(reviews, keywords):
    labels = []
    for review in reviews:
        if any(keyword in review.lower() for keyword in keywords):
            labels.append("Explicit")
        else:
            labels.append("Not Explicit")
    return labels

    
df['label'] = label_reviews(df['reviews'], explicit_words)

# Filtering explicit reviews and count them by book title
explicit_counts = df[df['label'] == 'Explicit'].groupby('book_title').size()

# Converting to DataFrame and resetting the index for plotting
explicit_counts = explicit_counts.reset_index(name='Explicit_Count')

# Sorting by 'Explicit_Count' for top 10 books
top_explicit_counts = explicit_counts.sort_values(by='Explicit_Count', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='book_title', y='Explicit_Count', data=explicit_counts, palette='Reds', ax=ax)

ax.set_xlabel('')
ax.set_ylabel('Number of Explicit Reviews')
ax.set_title('Explicit Reviews per Book Title')
ax.set_xticks([])  # X-axis labels were removed because 
st.pyplot(fig)

st.caption("Shows the number of explicit reviews associated with each book.")

##################################
