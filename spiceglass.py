### STREAMLIT APP ### 

import streamlit as st

# Streamlit page set up        
search_page = st.Page(
    page = "pages/Search.py",
    title = "Search",
    icon = ":material/search:", 
    default = True,
)

visuals_page = st.Page(
    page = "pages/Visuals.py",
    title = "Visuals",
    icon = ":material/bar_chart:",
)

contact_page = st.Page(
    page = "pages/Contact.py",
    title = "Contact",
    icon = ":material/email:",
)

pg = st.navigation(pages=[search_page, visuals_page, contact_page])

# Putting the logo in the corner
st.logo("assets/spiceglass_logo_r.png")

pg.run()

#######################################
