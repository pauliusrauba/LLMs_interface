import os
import json
import pandas as pd
from datetime import datetime
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
import sys
import numpy as np

# Import necessary functions and configurations
from openai_config import get_openai_config
from custom_tools import (
    get_qrisk3_information, calculate_ap_score, get_infosheet,
    get_nice_guidelines, plot_feature_importance_heart_risk,
    counterfactual_CVD_risk_ap, calculate_Qrisk_score, get_feature_importance
)

config = get_openai_config()

# FILES
QUESTION_FILE = './resources/documents/Questions_by_topic.xlsx'

# *** Include 'calculate_ap_score' in the TOOL_LIST ***
TOOL_LIST = [get_infosheet, get_nice_guidelines, calculate_ap_score, get_feature_importance, counterfactual_CVD_risk_ap]

# Tool descriptions for logging
tool_names = {
    'calculate_Qrisk_score': 'a method that calculates the QRisk3 score for a patient',
    'get_nice_guidelines': 'a document titled Cardiovascular disease: risk assessment and reduction, including lipid modification. This guideline provides a systematic approach for assessing cardiovascular disease (CVD) risk and offers recommendations for risk reduction, covering when to conduct formal risk assessments, how to use tools like QRISK3, and strategies for lifestyle changes, lipid management, and statin use to lower CVD risk in primary care settings.',
    'get_qrisk3_information': 'a document titled Development and validation of QRISK3 risk prediction algorithms to estimate future risk of cardiovascular disease',
    'counterfactual_CVD_risk_ap': 'a Counterfactual CVD risk estimation method',
    'get_infosheet': 'This QRISK3 infosheet provides a comprehensive overview of the cardiovascular risk prediction model, including its methodology, a breakdown of traditional and additional risk factors, detailed validation results, and performance metrics.',
    'get_feature_importance': 'a method to get feature importance for heart risk',
    'calculate_ap_score': 'a method that calculates the Autoprognosis cardiovascular disease risk score for a person using their data. It returns a detailed interpretation of the risk level.'
}

def create_llm():
    """Create and configure the Azure OpenAI LLM."""
    return AzureChatOpenAI(
        openai_api_base=config['api_base'],
        openai_api_version=config['api_version'],
        deployment_name=config['deployment_id'],
        openai_api_key=config['api_key'],
        openai_api_type=config['api_type'],
        temperature=0  # Set deterministic component
    )

def create_agent_chain(llm):
    """Create an agent chain with RAG components."""
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='output')
    tools = TOOL_LIST

    system_message = SystemMessage(content="""Assistant is an engaging, fun, verbose large language model which is an expert in medicine and machine learning, always asking whether any additional analyses should be run.
Assistant is kind and provides a lot of information. Assistant always provides ALL information that it observes. If the information is not provided in the info sheet, the assistant will also respond with its own knowledge.

Overall, Assistant is nice, always inquisitive, and asks for clarifications on whether any additional analyses should be run. Assistant always provides ALL information that it observes.""")

    agent_chain = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True, memory=memory, return_intermediate_steps=True,
        system_message=system_message
    )
    
    return agent_chain

def create_simple_chain(llm):
    """Create a simple LLMChain without RAG components."""
    template = """You are a helpful AI assistant with expertise in medicine and machine learning. 
Please respond to the following question or request:

Human: {question}
Assistant:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return LLMChain(llm=llm, prompt=prompt)

def get_responses(question):
    llm = create_llm()
    agent_chain = create_agent_chain(llm)
    simple_chain = create_simple_chain(llm)

    # Get response with RAG
    rag_response = agent_chain({"input": question})
    rag_answer = rag_response['output']
    intermediate_steps = rag_response['intermediate_steps']

    # Get response without RAG
    no_rag_response = simple_chain.run(question=question)

    return rag_answer, no_rag_response, intermediate_steps

def save_conversation(question_data, rag_answer, no_rag_answer, intermediate_steps):
    # Create the conversations directory if it doesn't exist
    output_dir = "./conversations/"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"conversation_{timestamp}.json")

    # Prepare the data to be saved
    data = {
        "higher_level_question": question_data["higher_level"],
        "lower_level_question": question_data["lower_level"],
        "rag_answer": rag_answer,
        "no_rag_answer": no_rag_answer,
        "rag_info": [],
        "intermediate_steps": []
    }

    if intermediate_steps:
        for step in intermediate_steps:
            action = step[0]
            observation = step[1]
            
            step_data = {
                "action": action.tool,
                "action_input": action.tool_input,
                "observation": observation,
                "thought": action.log,
                "tool_description": tool_names.get(action.tool, "Unknown tool")
            }
            
            data["intermediate_steps"].append(step_data)
            data["rag_info"].append(f"The model extracted information using {tool_names.get(action.tool, 'an unknown tool')}.")

    # Save the data to a JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Conversation saved to {filename}")
    return data

def save_summary(all_results):
    """Save summary of all results to CSV and JSON files."""
    output_dir = "./conversations"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results to JSON
    json_filename = os.path.join(output_dir, f"all_results_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to {json_filename}")

    # Save summary to CSV
    summary_data = [{
        "Higher-level question": result["higher_level_question"],
        "Lower-level question": result["lower_level_question"],
        "RAG_Answer": result["rag_answer"],
        "NO_RAG_Answer": result["no_rag_answer"],
        "RAG_Info": result["rag_info"]
    } for result in all_results]

    summary_df = pd.DataFrame(summary_data)
    csv_filename = os.path.join(output_dir, f"summary_results_{timestamp}.csv")
    summary_df.to_csv(csv_filename, index=False)
    print(f"Summary results saved to {csv_filename}")

def main():
    # Check if index is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide the index of the higher-level question as a command-line argument.")
        print("Usage: python script_name.py <higher_level_index> [<lower_level_index>]")
        return

    try:
        selected_higher_index = int(sys.argv[1])
    except ValueError:
        print("Invalid higher-level index provided. Please enter a valid integer.")
        return

    # Initialize selected_lower_index
    selected_lower_index = None
    if len(sys.argv) >= 3:
        try:
            selected_lower_index = int(sys.argv[2])
        except ValueError:
            print("Invalid lower-level index provided. Please enter a valid integer.")
            return

    # Load questions from Excel file
    try:
        questions = pd.read_excel(QUESTION_FILE)
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return

    # Get unique higher-level questions
    unique_higher_level_questions = questions['Higher-level question'].unique()
    total_higher_level_questions = len(unique_higher_level_questions)

    if selected_higher_index < 0 or selected_higher_index >= total_higher_level_questions:
        print(f"Invalid higher-level index. Please provide an index between 0 and {total_higher_level_questions - 1}.")
        return

    # Get the selected higher-level question
    selected_higher_level_question = unique_higher_level_questions[selected_higher_index]
    print(f"Selected higher-level question [{selected_higher_index}]: {selected_higher_level_question}")

    # Filter questions for the selected higher-level question
    selected_questions = questions[questions['Higher-level question'] == selected_higher_level_question]
    total_questions = len(selected_questions)

    if total_questions == 0:
        print("No questions found for the selected higher-level question.")
        return

    # Reset the index of selected_questions to 0..n-1
    selected_questions = selected_questions.reset_index(drop=True)

    # Print available lower-level questions and their indices
    print("Available lower-level questions and their indices:")
    for idx, question in selected_questions['Lower-level question'].iteritems():
        print(f"[{idx}]: {question}")

    if selected_lower_index is not None:
        if selected_lower_index < 0 or selected_lower_index >= total_questions:
            print(f"Invalid lower-level index. Please provide an index between 0 and {total_questions - 1}.")
            return
        # Select only the specified lower-level question
        selected_questions = selected_questions.iloc[[selected_lower_index]]
        print(f"Selected lower-level question [{selected_lower_index}]: {selected_questions.iloc[0]['Lower-level question']}")
    else:
        print("Processing all lower-level questions under the selected higher-level question.")

    # Initialize lists to store results
    all_results = []

    # Iterate over the filtered questions
    for index, row in selected_questions.iterrows():
        higher_level_question = row['Higher-level question']
        question = row['Lower-level question']

        print(f"Processing question: {question}")

        try:
            rag_answer, no_rag_answer, intermediate_steps = get_responses(question)
            question_data = {
                "higher_level": higher_level_question,
                "lower_level": question
            }
            result = save_conversation(question_data, rag_answer, no_rag_answer, intermediate_steps)
            all_results.append(result)
            print("Successfully processed question")
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            error_result = {
                "higher_level_question": higher_level_question,
                "lower_level_question": question,
                "rag_answer": "error",
                "no_rag_answer": "error",
                "rag_info": ["error"]
            }
            all_results.append(error_result)

    # Save summary files
    save_summary(all_results)
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
