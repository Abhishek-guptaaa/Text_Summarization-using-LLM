import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Load environment variables from the .env file
load_dotenv()

# Access the API key from the environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize the model
model = ChatGroq(api_key=api_key, model_name="gemma-7b-it", temperature=0.0)

# Define the prompt template
template = """
%INSTRUCTIONS:
Please summarize the following piece of text.
make it that easy so that a 5 year old would understand.

%TEXT:
{text}
"""

# Create the Streamlit app interface
st.title("Text Summarization for Kids")
st.write("Enter the text you want to summarize:")

# Text input from the user
summarisation_text = st.text_area("Input Text", height=200)

# Create a button to trigger summarization
if st.button("Summarize"):
    # Check if the user has entered text
    if summarisation_text:
        # Create the prompt
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template,
        )
        final_prompt = prompt.format(text=summarisation_text)

        # Create a HumanMessage object
        messages = [HumanMessage(content=final_prompt)]

        # Send the message to the model and get the summary
        output = model.invoke(messages)

        # Print only the summarized content
        st.subheader("Summary:")
        st.write(output.content)
    else:
        st.warning("Please enter text to summarize.")
