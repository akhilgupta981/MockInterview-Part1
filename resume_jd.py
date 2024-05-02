from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import chains
from langchain_core import prompts
from langchain_openai import ChatOpenAI
from IPython.display import Audio
from st_audiorec import st_audiorec
import requests
import streamlit as st
import os 
import PyPDF2


# load API keys for the models that will be used
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HF_KEY")

#Transcribe the recorded response to text
def speech2text(audio):

    speech2text = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    transcript = speech2text(audio)

    print("Output of speech2text is:")
    print(transcript)
    return transcript

#LLM function to use ChatGPT for generating feedback
def generate_feedback(response):
    
    template ="""
    You are an interview coach.
    Given a question and a candidate's response to the question, you provide critical feedback on their response.
    Do not use the word "feedback" in the beginning of your response and provide it in second person.
    The feedback should be limited to no more than 100 words.

    CONTEXT: {response}
    
"""

    prompt = prompts.PromptTemplate(template = template, input_variables =["response"])
    story_llm = chains.LLMChain(llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo", temperature = 0.5, max_tokens = 100), prompt = prompt, verbose = True)

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

    
# Generates question using ChatGPT given resume and jd file as inputs
def generate_question(resume, jd):
    # Read the content of the first file
    with open(resume, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        content1 = '\n'.join([t for t in text if t])
    
    
    # Read the content of the second file
    with open(jd, 'r', encoding='utf-8') as file:
        content2 = file.read()
    
    # Combine the content to form a prompt (customize as needed)
    combined_content = f"The contents of their resume are:\n{content1}\n\nAnd the job posting for the role that they are applying to is:\n\n{content2}\n\n"

    template ="""

    Ask a relevant interview question for the candidate who is interviewing for a job.
    CONTEXT: {combined_content}
    
    """

    # Use ChatGPT via langchain
    prompt = prompts.PromptTemplate(template = template, input_variables =["combined_content"])
    llm_chain = chains.LLMChain(llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo", temperature = 0.8, max_tokens = 50), prompt = prompt, verbose = True)

    question = llm_chain.predict(combined_content=combined_content)

    print("output of generate question function:")
    print(question)
    return question

# Saves uploaded files locally
def save_file(uploaded_file):
    with open(f'temp_{uploaded_file.name}', 'wb') as f:
        f.write(uploaded_file.getvalue())  # Save file to disk
    return f.name

def main():
    
    #Set up the page
    st.set_page_config(page_title="Mock Interviewer", page_icon=":memo:")
    st.header("Answer an interview question to get feedback on your response")

    # Allow user to upload resume and job description files
    resume = st.file_uploader("Upload Resume", type=['pdf'])
    jd = st.file_uploader("Upload Job Description", type=['pdf', 'txt', 'doc'])

    uploaded_file = None  # Initialize uploaded_file variable

    if resume and jd: #Condition to check if resume and jd have been uploaded
        resume_path = save_file(resume)
        jd_path = save_file(jd)

        if st.button("Generate Question"):
            question = generate_question(resume_path, jd_path) #Generate question
            st.session_state.question = question  # Save the generated question to session state
            st.write("Please record your response to the following question:")
            st.write(question)
            uploaded_file = st_audiorec()  # Recording response
        elif 'question' in st.session_state:
            # Display the question if already generated
            st.write("Please record your response to the following question:")
            st.write(st.session_state.question)
            uploaded_file = st_audiorec()  # Recording response

        

        # Processing after recording
        if uploaded_file is not None:
            with open('output_file.wav', 'wb') as f:
                f.write(uploaded_file)

            response = speech2text('output_file.wav')
            with st.expander("Response"):
                st.write(response)

            # Ensure that 'question' is fetched from session state
            if 'question' in st.session_state:
                analyze = f"The question is: {st.session_state.question} The candidate's response to the question is: {response['text']}"
                feedback = generate_feedback(analyze)
                with st.expander("Feedback in Text"):
                    st.write(feedback)
            else:
                st.error("Question data is missing. Please generate the question again.")
    else:
        st.write("Please upload the files above.")



if __name__ == '__main__':
    main()
