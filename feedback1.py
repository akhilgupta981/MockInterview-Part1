from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import chains
from langchain_core import prompts
from langchain_openai import ChatOpenAI
from IPython.display import Audio
from audiorecorder import audiorecorder
from st_audiorec import st_audiorec
import requests
import streamlit as st
import os 

# load API keys for the models that will be used
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HF_KEY")

#LLM function to use ChatGPT for generating text response
def generate_story(response):
    template ="""
    You are an interview coach.
    Given a question and a candidate's response to the question, you provide critical feedback on their response.
    Do not use the word "feedback" in the beginning of your response and provide it in second person.
    The feedback should be limited to no more than 50 words.

    CONTEXT: {response}
    
"""

    prompt = prompts.PromptTemplate(template = template, input_variables =["response"])
    story_llm = chains.LLMChain(llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo", temperature = 1, max_tokens = 50), prompt = prompt, verbose = True)

    story = story_llm.predict(response=response)

    print("output of llm function:")
    print(story)
    return story
 

#Convert text to speech to hear feedback as audio

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

#Transcribe the recorded response to text

def speech2text(audio):

    speech2text = pipeline("automatic-speech-recognition", model="openai/whisper-base")

    transcript = speech2text(audio)

    print("Output of speech2text is:")
    print(transcript)
    return transcript


#host on a UI

def main():

    #Page setup
    st.set_page_config(page_title="img 2 audio story", page_icon="Test")
    st.header("Answer an interview question to get feedback on your response")

    # Display a text input box where user can enter their own question
    question = st.text_input(label="What is the question you want to practice?", 
                             value="Why are you interested in this role?", max_chars= 100)

    # Initiate a button click counter
    if 'question_confirmed' not in st.session_state:
        st.session_state.question_confirmed = 0

    # Increment the button click counter when the button is clicked
    def on_question_confirmed():
        st.session_state.question_confirmed += 1

    # Records the user confirming the question they want to answer
    confirmquestion = st.button("Confirm Question", on_click=on_question_confirmed)    

    #initialize uploaded_file variable for audio recording
    uploaded_file = None 

    # Allow user to record their response once they have confirmed the question
    if st.session_state.question_confirmed > 0:
        st.write("Please record your response below.")        
        uploaded_file = st_audiorec()        
    else:
        st.write("Please confirm the question above")
  
    
    # Once an audio response has been recorded, the following logic is executed
    if uploaded_file is not None:           
    
        # Save the recorded audio to a file
        with open('output_file.wav', 'wb') as f:
            f.write(uploaded_file)
        
        
        response = speech2text('output_file.wav')  #translate the audio to text
        analyze = "The question is: " + question + "The candidate's response to the question is: " + response['text']  # Prepare the input for LLM function
        feedback = generate_story(analyze) # Get ChatGPT to provide feedback to the response
        verbalfeedback = text2speech(feedback) # Convert feedback to audio

        with st.expander("Response"): # Display the transcribed response
            st.write(response)
        with st.expander("Feedback in Text"): # Display the feedback to the response
            st.write(feedback)
        with st.expander("Feedback in Audio"): # Allow the user to play the feedback as audio
            st.audio(verbalfeedback)    

        
    else: 
        pass
    
    return
         
        
        
if __name__ == '__main__':
    main()


