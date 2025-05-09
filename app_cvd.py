import os
import pandas as pd
import numpy as np
import streamlit as st
from langchain.chat_models import AzureChatOpenAI

# Import streamlit apps
from streamlit_chat import message as st_message
from streamlit.components.v1 import html

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from openai_config import get_openai_config
from model_utils import manual_ap2_model_convert, calib_function

from custom_tools import get_qrisk3_information, calculate_ap_score,  get_infosheet, get_nice_guidelines, get_feature_importance, plot_feature_importance_heart_risk, counterfactual_CVD_risk, df_to_string, calculate_Qrisk_score

config = get_openai_config()

# Set OpenAI api key

# Prompt templates
template = """Assistant is an engaging, fun, verbose large language model which is an expert in medicine and machine learning, always asking for whether any additional analyses should be run.
Assistant is kind and provides a lot of information. Assistant always provides ALL information that it observes. 
Assistant always provides suggestions on what to evaluate next or what tool to use. 
The newest methods for cardiovascular risk predictions are the Qrisk scores. The Qrisk scores are the newest and most accurate methods for cardiovascular risk predictions.

Overall, Assistant is nice, always inquisitive and asks for clarifications on whether any additional analyses should be run. Assistant always provides ALL information that it observes. 
 """

@st.cache_resource()
def get_excel():
    df = st.session_state.df
    if df is not None:
        return df.to_excel()

@st.cache_resource()
def create_agent_chain():
    from model_utils import manual_ap2_model_convert, calib_function

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key='input', output_key="output")
    tools = [calculate_Qrisk_score, get_infosheet, calculate_ap_score, plot_feature_importance_heart_risk, get_feature_importance, get_nice_guidelines, get_qrisk3_information, counterfactual_CVD_risk]

    llm = AzureChatOpenAI(
        openai_api_base = config['api_base'],
        openai_api_version = config['api_version'],
        deployment_name = config['deployment_id'],
        openai_api_key = config['api_key'],
        openai_api_type = config['api_type'],
        temperature = 0)
    #llm = ChatOpenAI(openai_api_key=apikey, temperature=0, max_tokens=2500, model_name='gpt-3.5-turbo') #model_name='gpt-4' 
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                verbose=True, memory=memory, return_intermediate_steps=True,
                              input_key='input', output_key="output")
    
    agent_chain.agent.llm_chain.prompt.messages[0].prompt.template = template
    return agent_chain

tool_names = {
    'calculate_Qrisk_score': 'a method that calculates the QRisk3 score for a patient',
    'get_nice_guidelines': 'a document titled Cardiovascular disease: risk assessment and reduction, including lipid modification. Press [here](https://www.nice.org.uk/guidance/ng238) for the original document',
    'get_qrisk3_information': 'a document titled Development and validation of QRISK3 risk prediction algorithms to estimate future risk of cardiovascular disease. Press [here](https://doi.org/10.1136/bmj.j2099) for the original document',
    'counterfactual_CVD_risk': 'a Counterfactual CVD risk estimation method',
    'get_infosheet': 'Infosheet for the model',
    'plot_feature_importance_heart_risk': 'plot_feature_importance_heart_risk',
    'calculate_ap_score': 'calculate_ap_score',
    'get_feature_importance': 'get_feature_importance'

}
def generate_answer():
    input_text = st.session_state.input_text
    agent_chain = create_agent_chain()
    response = agent_chain({"input":input_text})
    print(response)
    # Get tools used
    msg_response = response['output']
    if response['intermediate_steps'] != []:
        tool_used = response['intermediate_steps'][0][0].tool
        tool_input = response['intermediate_steps'][0][0].tool_input

        st.session_state.history.append({"message": input_text, "is_user": True})
        st.session_state.history.append({"message": msg_response, "is_user": False, "info": f"The model extracted the information from {tool_names[tool_used]}."})
    
        if tool_used == 'plot_feature_importance_heart_risk':
            # Import feature importance plot from feature_importance.txt
            with open('./resources/patient_info/shap_values_john.txt', 'r') as f:
                html_img = f.read()

            st.session_state.history.append({"message": html_img, "is_user": False, "info": f"The model used the tool: {tool_used} with the following: {tool_input}. The description of the tool is: Use this for any question related to plotting the feature importance of heart risk for any patient or any model. The input should always be an empty string and this function will always return a tuple that contains the top three risk and their associated scores. It will always plot of feature importances. "})            
    else:
        st.session_state.history.append({"message": input_text, "is_user": True})
        st.session_state.history.append({"message": msg_response, "is_user": False})

# Initialize streamlit history

# Temporary function for demo purposes only. Quick workaround to see the UI interface.
def hide_code():
    st.markdown("""
        <style>
        code {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)
    
if "history" not in st.session_state:
    st.session_state.history = [{'message': 'Hello, I am the Medical AI assistant. How can I help you today?', 'is_user': False}]

st.title('🧠 💊 Medical Assistant Demo')
st.text_input('Start the conversation with the Medical AI assistant below.', key='input_text', on_change=generate_answer)

if "df" not in st.session_state:
    st.session_state.df = None

file = st.file_uploader("Upload relevant patient data", type=['csv'])

if file:
    st.session_state.df = pd.read_csv(file)
# 👩‍⚕️🤖
for i, chat in enumerate(st.session_state.history):
    hide_code()

    # If info exists in the chat, display the message in two columns (one for the message, one for the info).
    if 'info' in chat:
        col1, col2 = st.columns([5, 1])  
        col2.write(st_message(chat['message'], is_user=chat['is_user'], allow_html=True, key=str(i))) 

        # Show more information when checkbox is checked
        if col1.checkbox('Show source information', key=f"info_btn_{i}"):
            st.info(chat['info'])
        else:
            st.write("")  # This line is need
    # Otherwise, display the message normally.
    else:
        st_message(chat['message'], is_user=chat['is_user'], allow_html=True, key=str(i))
