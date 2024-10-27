import streamlit as st

st.title("Contact us!")

st.markdown("Below you can fill in a form if you have any questions or complaints. You may also use the message box to suggest any books we should add to the database.")

# Adding in the contact form 
with st.form("contact_form"):
    name = st.text_input("Name", placeholder="Your name")
    email = st.text_input("Email", placeholder="Your email")
    message = st.text_area("Message", placeholder="Type your message here")
    button = st.form_submit_button("Submit", use_container_width=True)
    
# This doesn't currenty connec to anything