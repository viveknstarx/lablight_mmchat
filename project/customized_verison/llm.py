import os
import gc
import torch
import logging
from dotenv import load_dotenv, find_dotenv, set_key
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
# from langchain_community.llms import Ollama
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

logging.basicConfig(level=logging.info)
logger = logging.getLogger("LLM.PY_FILE")

class MultiModelChatbot:
   
    """
    A class to initialize and manage a multimodal chatbot.
    Loads environment variables, creates vector database if new documents are given,
    and answers the questions using the built chatbot.
    """
   
  
    def __init__(self):
        load_dotenv(find_dotenv())

    def initialize_embeddings(self,embedding_model_path):
        """
        Initialize the embedding model. 
        Returns the embedding parameter which has intialized embedding model
        """
        global embedding
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        embedding = HuggingFaceEmbeddings(model_name=embedding_model_path, model_kwargs={"device": device})
        logger.info("Initializing embeddings on device: %s", device)
        return embedding

    def initialize_llm(self,inference_model_name,temp,max_tokens,repetition_penality,top_k,top_p):
        """
        Assuming you have Ollama installed and have llama3 model pulled with ollama pull llama3 .Intializes the inference model
        """
        global llm
        logger.info("Initializing LLM with model name: %s", inference_model_name)
        llm = Ollama(
            model= inference_model_name,
            temperature=temp, 
            num_ctx = max_tokens,
            repeat_penalty = repetition_penality,
            top_k = top_k, 
            top_p = top_p,
            keep_alive=0
            ) 
        # llm = ollama()
        return llm
    
    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:

        """
        Get the session history if available, otherwise initialize a new session history.
        """
       
        if session_id not in store_history:
            store_history[session_id] = ChatMessageHistory()
            logger.info("Initialized new session history for session_id: %s", session_id)
        else:
           
           keys = list(store_history.keys())
           items = list(store_history.values())
           session_id = keys[0]
           logger.info("Found existing session history for session_id: %s", session_id)
           store_history[session_id]=items[0]
        print("history:",history_length,"store_history:",store_history)
        if len(store_history[session_id].messages) > history_length:
            store_history[session_id].messages =store_history[session_id].messages[-history_length:]
        
        return store_history[session_id]

    def load_existing_db(self, vb_path,search_threshold,top_docs):
       
        """
        Load the vector database using the provided path and return it as a retriever.
        """
        db = Chroma(persist_directory=vb_path, embedding_function=embedding)
        search_kwargs = {"k": top_docs, "score_threshold":search_threshold}
        logger.info("Loading vector database from path: %s", vb_path)
        return db.as_retriever(search_kwargs=search_kwargs, search_type="similarity_score_threshold")

    def create_chain(self, rag_chain):
        
        """
        Create a chain with message history.
        """
        logger.info("Creating chain with message history.")
        return RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
       

    def setup_persona(self,vb_path,search_threshold,top_docs,prompt):
        
        """
        sets up persona for chatbot
        Accepts as vectordb path to load the vectordatabase
        search_threshold and top_docs for setting the retriver perameters
        prompt to set up the instructions for the chatbot to follow.

        """
        retriever = self.load_existing_db(vb_path,search_threshold,top_docs)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = (
            f"\nPersona: {prompt}\n\nContext: {{context}}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    
    def bot_creation(self,embedding_model_name,inference_model_name,vb_path,search_threshold,persona,instructions,top_docs,temp,max_tokens,repetition_penalty,top_k,top_p,store,input_text,version,historylength) -> str:
        """
        creates chatbot with given configuration details.
        Accepts all the configurations details which user has selected
        Returns the store_history and answer generated by chatbot
        """
        try:
            
            global store_history,history_length
            history_length = historylength
            store_history = store  
            prompt = persona+"\n"+instructions
            print(prompt)
            llm_model_path = os.getenv(inference_model_name)
            embedding_model_path = os.getenv(embedding_model_name)
            
            # keys = list(store_history.keys())
            session_id = version
           
            self.initialize_embeddings(embedding_model_path)
            self.initialize_llm(llm_model_path,temp,max_tokens,repetition_penalty,top_k,top_p)
            rag_chain = self.setup_persona(vb_path,search_threshold,top_docs,prompt)

            global main_chain
            main_chain = self.create_chain(rag_chain)
           
            response = main_chain.invoke(
                {"input": input_text},
                config={"configurable": {"session_id": session_id}}
            )
        
            response = response['answer']
            history = {version:store_history[version]}
            return  history,response
        except Exception as e:
            logger.error("Error during bot creation: %s", e)
            return store_history,e

       
    def generate_answer(self,input_text,version):
        """
        generates the answer if bot is built.
        Accepts the input text as congurations are same as previous question and bot is built already
        Returns the store_history and answer generated by chatbot
        """
        logger.info("Generating answer for session_id: %s", version)
        try:
            session_id = version
            response = main_chain.invoke(
                {"input": input_text},
                config={"configurable": {"session_id": session_id}}
            )
            
            response = response['answer']
            history = {version:store_history[version]}
            return  history,response
        except Exception as e:
            logger.error("Error during answer generation: %s", e)
            return e

    def clear_space(self):
        """
        cleares the cache .
        """
        try:
            global main_chain, embedding, llm
            flag =False
            # Safely delete the global variables if they exist
            if 'main_chain' in globals():
                del main_chain
                print("deleted chain")
            if 'embedding' in globals():
                del embedding
                print("deleted embedding")
            if 'llm' in globals():
                del llm
                flag = True
                print("deleted llm")
            if flag:
                logger.info("Clearing cache and releasing GPU memory.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Cleared cache and releasing GPU memory.")
        except:
            print("Nothing to clear the cache or gpu.")
            return None

       