from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import chains
from langchain_core import prompts
from langchain_openai import ChatOpenAI
from IPython.display import Audio
import requests
import streamlit as st
import os 

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HF_KEY")

#img2text
def img2text(url):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = img_to_text(url)[0]["generated_text"]

    print("output of img2text function:")
    print(text)
    return text

#llm
def generate_story(scenario):
    template ="""
    You are a storyteller.
    You can generate a short story based on a simple narrative. The story should be no more than 20 words.

    CONTEXT: {scenario}
    STORY:
"""

    prompt = prompts.PromptTemplate(template = template, input_variables =["scenario"])
    story_llm = chains.LLMChain(llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo", temperature = 1, max_tokens = 50), prompt = prompt, verbose = True)

    story = story_llm.predict(scenario=scenario)

    print("output of llm function:")
    print(story)
    return story
 

#textToSpeech

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.wav', 'wb') as file:
        file.write(response.content)
    return
    
#host on a UI

def main():

    st.set_page_config(page_title="img 2 audio story", page_icon="Test")

    st.header("Turn an image into an audio story")
    uploaded_file = st.file_uploader("Choose an image...", type = "jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Tmage.', use_column_width = True)

        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio("audio.wav")

if __name__ == '__main__':
    main()



