import streamlit as st
from spiceglass import average_spiciness_per_book

image_path = "assets/spiceglass_logo_simple.png"

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