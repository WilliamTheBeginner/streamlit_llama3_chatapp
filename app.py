# app.py
# Run this Streamlit application with:
#   streamlit run app.py

import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

# Load environment variables
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Initialize OpenAI client for LLaMA 3
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# Configure the Streamlit app
st.set_page_config(
    page_title="LLaMA3 ChatApp",
    page_icon="🤖",
    layout="centered"
)


def llama_chat(prompt: str, temperature: float = 1.0, max_tokens: int = 1024) -> str:
    """
    Send a prompt to the LLaMA 3 model and return its response.
    """
    response = client.chat.completions.create(
        model="meta/llama3-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].message.content


# Main application layout and logic
def main():
    st.title("🌟 LLaMA3 ChatApp AI Agent 🤖")
    user_query = st.text_input(
        "Ask Meta's LLaMA 3 something…",
        placeholder="Type your question here…",
    )

    if st.button("Send"):
        if user_query:
            with st.spinner("🤖 Thinking…"):
                answer = llama_chat(user_query)
            st.markdown("**Response:**")
            st.write(answer)


# Execute the app
main()
