from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import CTransformers
from langchain_core.prompts import FewShotPromptTemplate
from transformers import AutoTokenizer, AutoModel
from langchain_community.llms import LlamaCpp
from langchain.tools import Tool
from langchain.agents import Agent,AgentExecutor
from langchain_experimental.generative_agents import GenerativeAgent,GenerativeAgentMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_google_genai import ChatGoogleGenerativeAI

import faiss
import torch
import yaml
import faiss
import numpy as np
import re
import json
import os
from sentence_transformers import SentenceTransformer
import prompt_pipeline_spec as PromptPipeline
import Utility as util




def extract_stages(pipeline_str):
    pipeline_data= yaml.safe_load(pipeline_str)
    stages = []
    #Extract additional attributes
    additionai_attributes ={key:value for key,value in pipeline_data.items() if key not in {"stages","jobs","steps"}}
    if "stages" in pipeline_data:
        # Extract stages if present
        stages = pipeline_data["stages"]
    elif "jobs" in pipeline_data:
        # Extract jobs if stages are not present
        for job in pipeline_data["jobs"]:
            stage_name = guess_stage_name(job)
            stages.append({"stage": stage_name, "jobs":[job]})
    elif "steps" in pipeline_data:
        # Determine stage and extract steps if stages and jobs are not present
        for step in pipeline_data["steps"]:
            stage_name = guess_stage_name(step)
            stages.append({"stage": stage_name, "steps":[step]})
       
    
    return stages,additionai_attributes

def extract_stage_names(pipeline_str):
    pipeline_data= yaml.safe_load(pipeline_str)
    stage_names = []
    #Extract additional attributes
    additionai_attributes ={key:value for key,value in pipeline_data.items() if key not in {"stages","jobs","steps"}}
    if "stages" in pipeline_data:
        for stage in pipeline_data['stages']:
            if 'stage' in stage:
                stage_names.append(stage['stage'])
    elif "jobs" in pipeline_data:
        # Extract jobs if stages are not present
        for job in pipeline_data["jobs"]:
            stage_name = guess_stage_name(job)
            stage_names.append(stage_name)
    elif "steps" in pipeline_data:
        # Determine stage and extract steps if stages and jobs are not present
        for step in pipeline_data["steps"]:
            stage_name = guess_stage_name(step)
            stage_names.append(stage_name)
           
    return stage_names


def guess_stage_name_from_display_name(step):
    # Check if 'displayName' is present in the step
    if 'displayName' in step:
        return step['displayName']
    if 'script' in step and isinstance(step['script'], str):
        # Extract displayName from script if available
        display_name_match = re.search(r'displayName:\s*(.+)', step['script'])
        if display_name_match:
            return display_name_match.group(1).strip()
    return None

def guess_stage_name_from_keywords(step):
    # Fallback to keyword matching if displayName is not found
    keywords = ["build", "test", "deploy", "release", "package"]
    for keyword in keywords:
        if keyword in step.lower():
            return keyword.capitalize()
    return "Unknown"

def guess_stage_name(step):
    # First try to extract displayName
    stage_name = guess_stage_name_from_display_name(step)
    if stage_name:
        return stage_name
    
    # Then try keyword matching as a fallback
    return guess_stage_name_from_keywords(step)

# Function to create embeddings for a given text
def get_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        embeddings = model(**inputs)
    return embeddings.last_hidden_state.mean(dim=1).squeeze().numpy()

# Query the FAISS Index Function
def query_faiss_index(few_shot_examples,index,query_text, k=2):
    query_embedding = get_embeddings(query_text).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    relevant_examples=[few_shot_examples[i] for i in indices[0]]
    return relevant_examples

def create_chain(relevant_examples):
    #Format relevant examples
    formatted_examples = [
    {
        "input":"Azure Pipeline Stage",
        "target": "Pipeline Specification",
        "example_input":example["example_input"],
        "example_output":example["example_output"]
    } for example in relevant_examples if example["input"]=="Azure Pipeline Stage" and example["target"]=="Pipeline Specification"
    ]


    #Define the FewShotTemplate using the retrieved examples
    example_prompt=PromptTemplate(template="""input:{input}\n target: {target}\n example_input:{example_input}\n example_output:{example_output}""",input_variables=["input","target","example_input"])
    template= FewShotPromptTemplate(
        examples=formatted_examples,
        example_prompt=example_prompt,
        prefix=PromptPipeline.pipelinePrompt_TRAI,
        suffix=" input: {input}\n target: {target}\n example_input:{example_input}\n example_output:",
        input_variables=["input","target","example_input"]
    )  
        # Define the FewShotTemplate using the retrieved examples
   
   
    # Define the chain function
    def chain(input_text):
         # Initialize the LlamaCpp model  
        
        os.environ["GOOGLE_API_KEY"]=""
        #llm = ChatGoogleGenerativeAI(
          #model="gemini-1.5-pro",
          #temperature=0,
          #max_tokens=None,
          #timeout=None,
          #max_retries=5
          # other params...
        #)
        #llm = ChatGoogleGenerativeAI(model="gemini_1.5_flash")
        #llm= OllamaLLM(model="llama3.2:3b",base_url="http://localhost:11434/")
        llm=util.load_model("gemini-2.0-flash")
        formatted_input = template.format(input="Azure Pipeline Stage",target="Pipeline Specification",example_input=input_text)
        return llm.invoke(formatted_input).content
    return chain

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Initialize the vectorstore as empty
    embedding_size = 384
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
           )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

def devops_agent_provide_inst(pipeline_details):
    """
    Provides guidance and instructions for handling the pipeline stages.

    Args:
        pipeline_details (str): The pipeline configuration.

    Returns:
        str: Instructions for improving pipeline efficiency.
    """
    instructions = f"""
    DevOps Guidelines for Pipeline Optimization:

    1. Validate correctness of pipeline structure for {pipeline_details}.
    2. Optimize execution time by parallelizing jobs where possible.
    3. Ensure security compliance with best DevOps practices.
    4. Implement version control to track modifications.
    5. Suggest improvements for better automation.

    Outcome: A well-optimized and validated pipeline.
    """
    
    return instructions


def pipeline_agent_finalize_spec(spec_details):
    """
    Finalizes the pipeline specification by making necessary adjustments.

    Args:
        spec_details (str): The initial generated pipeline specification.

    Returns:
        str: Finalized and refined pipeline specification.
    """
    # Simulate finalizing pipeline spec
    finalized_spec = spec_details + "\nFinal Review: Approved by DevOps Team"

    return f"Finalized Pipeline Specification:\n{finalized_spec}"

def devops_agent_provide_description():
    
    # Simulate providing a description of the source pipeline
    #description = "DevOps Agent: The source pipeline includes stages A, B, and C with tools X, Y, and Z."
    with open ('C:/Users/196311/Documents/GenAI/Input/azure-pipelines.yml','r') as file:
      pipeline = yaml.safe_load(file)
    #description="Create specification from following pipeline\n", pipeline
    return pipeline

def devops_agent_verify_summary(pipeline_summary):
    """
    Validates the correctness of the extracted pipeline summary.

    Args:
        pipeline_summary (str): The summary generated by the Pipeline Agent.

    Returns:
        str: Verification status.
    """
    # Simulate verification logic
    verification_status = "Pipeline summary verified successfully."

    return f"Verification Result:\n{verification_status}\nSummary:\n{pipeline_summary}"

def pipeline_agent_create_spec(azure_yaml):
    """
    Generates a pipeline specification from the given Azure YAML configuration.

    Args:
        azure_yaml (str): The Azure DevOps pipeline YAML file content.

    Returns:
        str: A structured pipeline specification.
    """
    spec= generate_spec_from_code(azure_yaml)
    spec_list=["Pipeline Agent: Created the pipeline specification.Please verify.", str(spec)]
    final_spec="\n".join(spec_list)
    print("final_spec ", final_spec)
    return final_spec

def devops_agent_verify_spec(spec_details):
    """
    Reviews and verifies the correctness of the pipeline specification.

    Args:
        spec_details (str): The generated pipeline specification.

    Returns:
        str: Verification status.
    """
    # Simulate verification process
    verified_spec = spec_details + "\nVerification Status: Passed all checks."

    return f"DevOps Agent:Verified Pipeline Specification:\n{verified_spec}"


def pipeline_agent_summarize(pipeline_details):
    """
    Extracts the stages from the given pipeline configuration.

    Args:
        pipeline_details (str): The pipeline specification or YAML structure.

    Returns:
        str: A structured list of extracted stages.
    """
 
    # Simulate summarizing the number of stages and tools used
    extracted_stages= extract_stage_names(pipeline_details)
    return f"Extracted stages from pipeline.Please verify:\n" + "\n".join(extracted_stages)

# Define the conversation flow
def conversation_flow(pipeline):
    conversation = []
    print("pipeline ", pipeline)
    #DevOps Agent provides the description of the source pipeline
   # pipeline = devops_agent_provide_description()
    description_list=["DevOps Engineer Agent:Please extract stages from the pipeline"]
    description ="\n".join(description_list)
    conversation.append(description)
 

    # Pipeline Agent summarizes the stages and tools used
    summary = pipeline_agent_summarize(pipeline)
    conversation.append(summary)

    # DevOps Agent verifies the summary
    summary_verification = devops_agent_verify_summary(summary)
    conversation.append(summary_verification)

    # Pipeline Agent creates the specification
    spec = pipeline_agent_create_spec(pipeline)
    conversation.append("\n")
    conversation.append(spec)

    # DevOps Agent verifies the specification
    spec_verification = devops_agent_verify_spec(spec)
    conversation.append(spec_verification)

    # Pipeline Agent finalizes the specification
    final_spec = pipeline_agent_finalize_spec()
    conversation.append(final_spec)

    return conversation


def generate_spec_from_code(pipeline_str,os):
    print("Operating System ", os)
    # Initialize the tokenizer and model for embeddings
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    with open ('C:/Users/RajatJana/Desktop/DevXcelerate/Examples/Example_A2S.json','r') as f:
      few_shot_examples = json.load(f)
    #Create embeddings for each stage
    pipeline_stages,additional_attributes=extract_stages(pipeline_str)
    #Generate embeddings for each stage description and use stage name as key
    stage_descriptions = {stage["stage"]:yaml.dump(stage) for stage in pipeline_stages}
    #stage_embeddings = {stage_name: get_embeddings(description) for stage_name,description in stage.description.items()}
    stage_embeddings = {stage_name:get_embeddings(description) for stage_name,description in stage_descriptions.items()}
    
    #Generate embeddings for FewShot examples
    example_embeddings = [get_embeddings(example["example_input"]) for example in few_shot_examples]

    #Create and Populate Faiss Index
    dimension = example_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    embedding_matrix = np.array(example_embeddings).reshape(-1,dimension)
    index.add(embedding_matrix)

    #Save example inputs for mapping results
    example_inputs = [example["example_input"] for example in few_shot_examples]

    #Combine all stages descriptions into a single input
    combined_stage_descriptions="\n".join(stage_descriptions.values())
    print("combined_stage_descriptions ",combined_stage_descriptions)
    #Query the FAISS index using the combined stage descriptions
    relevant_examples = query_faiss_index(few_shot_examples,index,combined_stage_descriptions)
    chain=create_chain(relevant_examples)
    #Generate specification using the chain
    specification_str = chain(combined_stage_descriptions)
    print("Specification String ",specification_str)
    #specification = json.loads(specification_str)
    #wrap the specification list in a dictionary
    #wrapped_specification = {
     #   "stages": specification
    #}
    #Add additional attributes to the specification
   # additional_attributes.update(wrapped_specification)
    #final_specification = {**additional_attributes,**specification}
    return specification_str
   
def initiate_conversation(pipeline_str):
    # Create tool instances
    stage_tool = Tool(name="Stages Extractor", func=pipeline_agent_summarize, description="Extract stages from a given Pipeline")
    generate_spec_tool = Tool(name="Spec Generator", func=pipeline_agent_create_spec, description="Generate pipeline specification from an Azure YAML")
    finalize_spec_tool = Tool(name="Spec Finalization", func=pipeline_agent_finalize_spec, description="Finalize the pipeline specification")

    # Create the memory retriever
    retrieve_memory = create_new_memory_retriever()

    # List of tools assigned to Pipeline Agent
    pipeline_tools = [stage_tool, generate_spec_tool, finalize_spec_tool]
    
    # Initialize the LLM model
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    #os.environ["GOOGLE_API_KEY"]="AIzaSyArYq9kel4SMUDYNpFhhuRq40YJQuICNwM"
    #llm = ChatGoogleGenerativeAI(
         # model="gemini-1.5-pro",
         # temperature=0,
         # max_tokens=None,
         # timeout=None,
         # max_retries=2,
          # other params...
        #)
    llm= OllamaLLM(model="llama3.2:3b",base_url="http://localhost:11434/")

    # Create memory for Pipeline Agent
    pipeline_agent_memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=retrieve_memory,
        verbose=False,
        reflection_threshold=8,
    )

    # Create Pipeline Agent
    pipeline_agent = GenerativeAgent(
        name="Pipeline Agent",
        llm=llm,
        tools=pipeline_tools,
        description="Handles DevOps Pipeline tasks",
        status="spec generation",
        max_tokens=150,
        temperature=0.2,
        top_p=0.9,
        memory=pipeline_agent_memory
    )

    # Create DevOps Agent memory
    devops_agent_memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=retrieve_memory,
        verbose=False,
        reflection_threshold=8,
    )

    # Create tool instances for DevOps Agent
    provide_inst_tool = Tool(name="Instruction Provider", func=devops_agent_provide_inst, description="Provide Instruction to Pipeline Agent")
    verify_summary_tool = Tool(name="Summary Verifier", func=devops_agent_verify_summary, description="Verify pipeline summary")
    verify_spec_tool = Tool(name="Spec Verifier", func=devops_agent_verify_spec, description="Verify pipeline specification")

    # List of tools assigned to DevOps Agent
    devops_tools = [provide_inst_tool, verify_summary_tool, verify_spec_tool]

    # Define the DevOps Agent
    devOps_agent = GenerativeAgent(
        name="DevOps Agent",
        llm=llm,
        tools=devops_tools,
        description="Handles DevOps tasks",
        status="DevOps task handling",
        max_tokens=150,
        temperature=0.2,
        top_p=0.9,
        memory=devops_agent_memory
    )

    # Execute conversation flow
    conversation = []

    # Initiate dialogue with Pipeline Agent
    pipeline_agent_input = f"Extract the stages from the given pipeline: {pipeline_str}"

    success, response = pipeline_agent.generate_dialogue_response(pipeline_agent_input)

    # Simulate the conversation loop
    for _ in range(4):
        print("Pipeline Agent:", response)
        conversation.append(f"Pipeline Agent: {response}")

        # Ensure DevOps Agent correctly verifies the pipeline summary
        devops_agent_input = f"Verify pipeline summary: {response} for input pipeline: {pipeline_str}"

        success, response = devOps_agent.generate_dialogue_response(devops_agent_input)

        print("DevOps Engineer Agent:", response)
        conversation.append(f"DevOps Engineer Agent: {response}")

        # Explicitly instruct DevOps Agent to use verification tools
        resp = devOps_agent.construct(devops_agent_input)
        print("DevOps Engineer Agent Construct:", resp)
        conversation.append(f"DevOps Engineer Agent Construct: {resp}")

        # Generate pipeline specification
        pipeline_agent_input = f"Use 'Spec Generator' to create pipeline specification for: {pipeline_str} with summary: {response}"
        success, response = pipeline_agent.generate_dialogue_response(pipeline_agent_input)
        
        print("Pipeline Agent:", response)
        conversation.append(f"Pipeline Agent: {response}")

        # Verify the specification using DevOps Agent
        devops_agent_input = f"Verify pipeline specification: {response}"
        success, response = devOps_agent.generate_dialogue_response(devops_agent_input)

        print("DevOps Engineer Agent:", response)
        conversation.append(f"DevOps Engineer Agent: {response}")

        # Finalize the pipeline specification
        pipeline_agent_input = f"Use 'Spec Finalization' to update the pipeline spec based on feedback: {response}"
        success, response = pipeline_agent.generate_dialogue_response(pipeline_agent_input)

        print("Pipeline Agent:", response)
        conversation.append(f"Pipeline Agent: {response}")

    # Final verification step
    final_verification = devOps_agent.generate_dialogue_response(f"Finalize this specification: {response}")
    
    print("DevOps Engineer Agent (Final):", final_verification)
    conversation.append(f"DevOps Engineer Agent (Final): {final_verification}")

    return conversation







