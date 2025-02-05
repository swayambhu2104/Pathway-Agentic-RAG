import os
import pathway as pw
from pathway.xpacks.llm import llms, embedders, prompts, parsers, splitters
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer, RAGClient, AdaptiveRAGQuestionAnswerer
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.udfs import ExponentialBackoffRetryStrategy
from pathway.udfs import DiskCache, ExponentialBackoffRetryStrategy 
from openai import OpenAI
import threading
from serpapi.google_search import GoogleSearch as search
from bs4 import BeautifulSoup
import requests
from llm import OpenAIClient
from scraper import ContentScraper, GoogleSerperAPI
from guardrail import GuardrailChecker
from conversational_agent import ConversationalPipeline
from grade import grade_doc
import config
import google.generativeai as genai
from config import key
import time
import os
import aiohttp
from typing import Any, Dict, List, Optional

# Set environment variables for API keys
cred = key("Enter SERP API KEY", "Enter GEMINI API KEY", "ENTER OPENAI API KEY", "ENTER PATHWAY LICENSE KEY", "ENTER SERPER API KEY")


# Initialize necessary components for processing text and generating responses
text_splitter = splitters.TokenCountSplitter(max_tokens=400)  
embedder = embedders.OpenAIEmbedder(cache_strategy=DiskCache())  

# Load data from the specified folder
folder = pw.io.fs.read(path="./data/", format="binary", with_metadata=True)
sources = [folder]

# Set up the LLM (Large Language Model) for response generation
chat = llms.OpenAIChat(  
    model="gpt-4o",
    retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),  # Retry strategy for failed requests
    cache_strategy=DiskCache(), # Caching to avoid redundant API calls
    temperature=0.05, # Set low temperature for consistent responses
)

table_args = {
    "parsing_algorithm": "llm",
    "llm": chat,
    "prompt": prompts.DEFAULT_MD_TABLE_PARSE_PROMPT,
}
parser = parsers.OpenParse(table_args=table_args)

# Set up document store with sources, embedder, splitter, and parser
doc_store = VectorStoreServer(
    *sources,
    embedder=embedder,
    splitter=text_splitter,
    parser=parser
)


# Initialize the Adaptive RAG (Retrieval Augmented Generation) question-answering system
app = AdaptiveRAGQuestionAnswerer(  
    llm=chat,
    indexer=doc_store,
)

# Define server host and port
app_host = "0.0.0.0"
app_port = 8000

def start_server():
    """
    Starts the Adaptive RAG Question Answerer server. 
    This method builds and runs the server in a separate thread.
    """
    app.build_server(host=app_host, port=app_port)
    app.run_server()


# Run the server in a separate thread to allow for real-time question answering
server_thread = threading.Thread(target=start_server, name="AdaptiveRAGQuestionAnswerer")
server_thread.daemon = True
server_thread.start()

import time
time.sleep(2)

# Create a client for interacting with the RAG server
client = RAGClient(host=app_host, port=app_port)

print("Server is running. You can now ask questions. Type 'exit' to stop.")
while True:
    print()
    guard = GuardrailChecker(cred.openai_api_key)  # Guardrail to check for inappropriate content
    grader = grade_doc(cred.openai_api_key)  # Grader for classifiying relevance of retrieved documents
    question = str(input("Enter your question: "))
    
    # Exit condition for the loop
    if question.lower() == "exit":
        break
    # response = client.pw_ai_answer(question)
    # print("Response:", response)
    
    # Check if the question is appropriate
    if(guard.check_compliance(question) == "no"):
        print("Inappropriate query")
        print(guard.generate_response(question))
        continue
      
    # Retrieve context from the RAG server
    docs = client.retrieve(question)
    texts = [item['text'] for item in docs]
    
    # Grade the retrieved context against the query
    status = grader.grade_document(question, texts)

    # agent_2 = OpenAIClient(OPENAI_API_KEY)



    leader_analyst = ConversationalPipeline(cred.openai_api_key) # Initialize leader_analyst conversational pipeline


    if(status.lower() == "yes"):
        print("Response: ", "Correct")
        print("Entering Leader-Analyst chain")


        # If the query can be answered accurately, divide the task into subtasks
        subtask_1, subtask_2 = leader_analyst.divide_correct_task_into_subtasks(question, texts)

        print(f"""Query : {question} divided into two subtasks\n
        Subtask_1 : {subtask_1}\n
        Subtask_2 : {subtask_2}""")

        # Retrieve context for each subtask
        context_a = client.retrieve(subtask_1)
        context_a = [item['text'] for item in context_a]

        context_b = []
        if(subtask_2!=""):
          context_b = client.retrieve(subtask_2)
          context_b = [item['text'] for item in context_b]

        # Run the leader-analyst pipeline to generate a response
        final_response = leader_analyst.run_pipeline(question, context_a, context_b, subtask_1, subtask_2)
        # print("Response: ", final_response)

        # Check if a follow-up task is required
        follow_up_status = leader_analyst.check_follow_up(question, context_a + context_b, final_response)
        print("Follow-up status: ", follow_up_status)

        
        # If follow-up is needed, further divide the query and retrieve additional context
        if(follow_up_status == "Yes" and subtask_2 != ""):  

          subtask_3, subtask_4 = leader_analyst.generate_new_subtasks(question, subtask_1, subtask_2, texts)

          print(f"""Query : {question} further divided into two more subtasks:\n
          Subtask_3 : {subtask_3}\n
          Subtask_4 : {subtask_4}""")

          context_c = client.retrieve(subtask_3)
          context_c = [item['text'] for item in context_c]

          context_d = client.retrieve(subtask_4)
          context_d = [item['text'] for item in context_d]


          context = []
          context.extend(context_a)
          context.extend(context_b)

          final_response = leader_analyst.run_pipeline_if_needed(question, context_c, context_d, subtask_3, subtask_4, final_response, context)

          print("Response: ", final_response)
          feedback = str(input("\n\nAre you satisfied with the response? \n\n Please answer Yes or No :\n"))
          if(feedback.lower()=="yes"):
            print("Query Resolved")
          else:
            print("Please enter the refined version of the query along with additional context")


        else:
          print("Response: ", final_response)
          feedback = str(input("\n\nAre you satisfied with the response? \n\n Please answer Yes or No :\n"))
          if(feedback.lower()=="yes"):
            print("Query Resolved")
          else:
            print("Please enter the refined version of the query along with additional context")

  

    else:
        # If the query cannot be answered accurately with the given documents, do web search to retrieve context
        status = "Incorrect"
        print("Response: ", status)
        print("Entering leader-analyst chain")
        print("Doing web-search to find the answer")
        
        # Use web scraping APIs to fetch additional information
        web_scraper = GoogleSerperAPI(cred.serper_api_key)
        flag = 1
        if(web_scraper.initialised and flag):
          print("\nUsing SERPER API FOR WEB SEARCH\n")
          subtask_1, subtask_2 = leader_analyst.divide_incorrect_task_into_subtasks(question)
          print(f"""Query : {question} divided into two subtasks\n
          Subtask_1 : {subtask_1}\n
          Subtask_2 : {subtask_2}""")

          context_a = web_scraper.search(subtask_1)
          context_b = ""
          if(subtask_2 != ""):
            context_b = web_scraper.search(subtask_2)
          
          # Run the pipeline with the web-scraped context
          final_response = leader_analyst.run_pipeline(question, context_a, context_b, subtask_1, subtask_2)
          
          context = []
          context.extend(context_a)
          context.extend(context_b)
          
          follow_up_status = leader_analyst.check_follow_up(question, context, final_response)
          print("Follow-up status: ", follow_up_status)
          
          # Handle follow-up queries
          if(follow_up_status == "Yes" and subtask_2 != ""):
            subtask_3, subtask_4 = leader_analyst.generate_new_subtasks(question, subtask_1, subtask_2, texts)
            print(f"""Query : {question} further divided into two more subtasks:\n
            Subtask_3 : {subtask_3}\n
            Subtask_4 : {subtask_4}""")
            
            context_c = web_scraper.search(subtask_3)
            context_d = ""
            if(subtask_4 != ""):
              context_d = web_scraper.search(subtask_4)
            
            final_response = leader_analyst.run_pipeline_if_needed(question, context_c, context_d, subtask_3, subtask_4, final_response, context)
            print("Response: ", final_response)
            # Ask the user for feedback on the generated response (Human in the Loop)
            feedback = str(input("\n\nAre you satisfied with the response? \n\n Please answer Yes or No: \n"))
            if(feedback.lower()=="yes"):
              print("Query Resolved")
            else:
              print("Please enter the refined version of the query along with additional context")
          else:
            print("Response: ", final_response)
            # Ask the user for feedback on the generated response (Human in the Loop)
            feedback = str(input("\n\nAre you satisfied with the response? \n\n Please answer Yes or No: \n"))
            if(feedback.lower()=="yes"):
              print("Query Resolved")
            else:
              print("Please enter the refined version of the query along with additional context")
                    
          
        else :  
          # Fallback to SERPER API
          print("SERPER API FAILED. FALLBACK INITIATED. USING SERP API FOR WEB SEARCH")
          
          # Initialize the web scraper with the SERP API key
          web_scraper = ContentScraper(cred.serp_api_key)
          
          # Divide the original incorrect query into two subtasks for further processing
          subtask_1, subtask_2 = leader_analyst.divide_incorrect_task_into_subtasks(question)

          print(f"""Query : {question} divided into two subtasks\n
          Subtask_1 : {subtask_1}\n
          Subtask_2 : {subtask_2}""")

           # Fetch the context for Subtask 1 and Subtask 2 from the SERP API by performing a Google search
          source_description_list_a, ai_overview_context_a = web_scraper.search_google(subtask_1)
          all_content_a, context_a = web_scraper.get_content_from_urls(source_description_list_a)
          stock_info_a = web_scraper.get_stock_price(subtask_1)

          source_description_list_b, ai_overview_context_b = web_scraper.search_google(subtask_2)
          all_content_b, context_b = web_scraper.get_content_from_urls(source_description_list_b)
          stock_info_b = web_scraper.get_stock_price(subtask_2)

          # Merge all relevant contexts (content and stock info) for both subtasks
          context_a.extend(ai_overview_context_a)
          context_a.extend(stock_info_a)
          context_b.extend(ai_overview_context_b)
          context_b.extend(stock_info_b)

          # Run the leader-analyst pipeline using the contexts for both subtasks
          final_response = leader_analyst.run_pipeline(question, context_a, context_b, subtask_1, subtask_2)
          # print("Response: ", final_response)

          
          # Combine the contexts for both subtasks into a single context list
          context = []
          context.extend(context_a)
          context.extend(context_b)
          follow_up_status = leader_analyst.check_follow_up(question, context, final_response)
          print("Follow-up status: ", follow_up_status)

          # If a follow-up is needed, further divide the query and retrieve additional context
          if(follow_up_status == "Yes" and subtask_2 != ""):
            subtask_3, subtask_4 = leader_analyst.generate_new_subtasks(question, subtask_1, subtask_2, texts)

            print(f"""Query : {question} further divided into two more subtasks:\n
            Subtask_3 : {subtask_3}\n
            Subtask_4 : {subtask_4}""")

            # Fetch content and stock information for the new subtasks
            source_description_list_c, ai_overview_context_c = web_scraper.search_google(subtask_3)
            all_content_c, context_c = web_scraper.get_content_from_urls(source_description_list_c)
            stock_info_c = web_scraper.get_stock_price(subtask_3)

            source_description_list_d, ai_overview_context_d = web_scraper.search_google(subtask_4)
            all_content_d, context_d = web_scraper.get_content_from_urls(source_description_list_d)
            stock_info_d = web_scraper.get_stock_price(subtask_4)

            # Merge the contexts and stock information for the new subtasks
            context_c.extend(ai_overview_context_c)
            context_c.extend(stock_info_c)
            context_d.extend(ai_overview_context_d)
            context_d.extend(stock_info_d)

            # Print the final response after the follow-up processing
            final_response = leader_analyst.run_pipeline_if_needed(question, context_c, context_d, subtask_3, subtask_4, final_response, context)

            print("Response: ", final_response)
            
            # Ask the user for feedback on the generated response (Human in the Loop)
            feedback = str(input("\n\nAre you satisfied with the response? \n\n Please answer Yes or No: \n"))
            if(feedback.lower()=="yes"):
              print("Query Resolved")
            else:
              print("Please enter the refined version of the query along with additional context")
          else:
            print("Response: ", final_response)
            # Ask the user for feedback on the generated response (Human in the Loop)
            feedback = str(input("\n\nAre you satisfied with the response? \n\n Please answer Yes or No: \n"))
            if(feedback.lower()=="yes"):
              print("Query Resolved")
            else:
              print("Please enter the refined version of the query along with additional context")
