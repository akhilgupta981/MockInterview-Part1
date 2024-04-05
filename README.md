Download the folder on your machine/environment from where you plan to run this

Install the libraries required to run the application. This is typically done using the Command line / Terminal
  streamlit: pip install streamlit
  langchain: pip install langchain
  ...etc

Create a .env file in the same folder to store your API keys. You need to retrieve your private keys by navitaging to settings in ChatGPT and HuggingFace respectively. 
Open the file in a text editor and save your APIs in the following format:
  OPENAI_API_KEY=sk****jyAZ
  HF_KEY=hf_****dNmV

Run the application from the command line using the following prompt: 
  streamlit run feedback1.py
