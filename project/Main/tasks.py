import os
import logging
from dotenv import load_dotenv, find_dotenv
from project.Main.util import make_rest_call_userbackend,q_and_a_payload
from project.customized_verison.llm import MultiModelChatbot
from project.customized_verison.embedding_creation import EmbeddingCreation
from project.customized_verison.history_from_txt import format_history

_ = load_dotenv(find_dotenv())
logging.basicConfig(level = logging.INFO)
logger =logging.getLogger("tasks.py")

class Tasks():
  def __init__(self) -> None:
     pass
  
  def bot_creation_with_configiration(self,embedding_model_name,inference_model_name,vb_path,search_threshold,persona,instructions,top_docs,temp,max_tokens,repetition_penalty,top_k,top_p,history,input_text,version,historylength,url):
      """
      A function to call the methods in MultiModalchatbot to create chatbot and gets the answer to the question asked by user
      post the response to the middle_ware.
      """
      try:    
          MultiModelChatbot(). clear_space()
          store = format_history(history) 
          store_history,response =  MultiModelChatbot().bot_creation(embedding_model_name,inference_model_name,vb_path,search_threshold,persona,instructions,top_docs,temp,max_tokens,repetition_penalty,top_k,top_p,store,input_text,version,historylength)
          make_rest_call_userbackend(url,q_and_a_payload(response,store_history))
          return response
      except Exception as e:
        print("Failed due to:",e)
        return e
  
  def  get_answer(self,embedding_model_name,inference_model_name,vb_path,search_threshold,persona,instructions,top_docs,temp,max_tokens,repetition_penalty,top_k,top_p,history,input_text,version,historylength,url):
      """
      calls the method generate answer from already built chatbot. if error occurs tries to build the chatbot and get the answer
      """
      
      try:
        store_history,response=MultiModelChatbot().generate_answer(input_text,version)
        
        make_rest_call_userbackend(url,q_and_a_payload(response,store_history))
        return response
      except:
         MultiModelChatbot(). clear_space()
         store = format_history(history) 
         store_history,response =  MultiModelChatbot().bot_creation(embedding_model_name,inference_model_name,vb_path,search_threshold,persona,instructions,top_docs,temp,max_tokens,repetition_penalty,top_k,top_p,store,input_text,version,historylength)
         make_rest_call_userbackend(url,q_and_a_payload(response,store_history))
        
         return response


  def create_customized_vb(self,document_path,vector_db_path,embedding_model_name,chunk_size,chunk_overlap):
      """
      calls the method create vector index to create embeddings and save them.
      """
      try:
        embedding_model_path = os.getenv(embedding_model_name)
        EmbeddingCreation().create_vector_index(document_path,vector_db_path,embedding_model_path,chunk_size,chunk_overlap)
      except Exception as e:
          print(e)
