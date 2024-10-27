# SPICEGLASS

## For picking up your next read... 

This app was developed as a part of a Introduction to Data Science course, 
with real users in mind. Spiceglass is a tool aimed at readers interested in the 
romance genre, to help give readers an overview of the amount of explicit and 
sexual content without any spoilers! Spiceglass utilizes Goodreads reviews, a platform 
users have utilized to share and rate the sexual nature of the books using their own rating 
system of chili-peppers. Seeing how users had created their own system for something that 
has been lacking on platforms, we decided to utilize and preserve the system that the community 
had built and create a tool with readers in mind. 

## How it works: 

The results are presented using streamlit. The user types in the title of the book and the app
gives it a spiciness rating of one to five chilli peppers, along with a short
description of what it means. Currently, the app only works with the 100 books 
that have been added to the database. For all other book titles, the app 
prints out the message “Sorry, [book title] hasn’t yet been added to our 
database.” 

## Data collection: 

We used RStudio’s Goodreader library to scrape 30 reviews of 100 books. 
The data was saved as a CSV file and exported to Python. The dataframe includes 
book id (book_id), the book title (book_title) and review text (reviews). 
No data was collected from copyrighted material. 

## How to use: 

1. Download the files to a directory 
2. Activate the python environment
3. Check the path to the dataframe on your computer and edit the code accordingly
4. To view streamlit on a browser, run the following command: streamlit run /filepath.py

