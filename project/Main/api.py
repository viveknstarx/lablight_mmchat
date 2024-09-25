import random
import string
import os
import json
from time import sleep
from pydantic import BaseModel
from dotenv import load_dotenv,find_dotenv
from redis import Redis
from fastapi import FastAPI, BackgroundTasks,HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from project.customized_verison.speechtotext import transcribe_audio_from_file
from project.Main.tasks import Tasks
from fastapi.responses import FileResponse
_ = load_dotenv(find_dotenv())

redis_host = os.getenv('redis_host','localhost')
redis_port = int(os.getenv('redis_port',6379))
redis_db = int(os.getenv('redis_db',0))
redis_response =bool(os.getenv('redis_response',True))
# redis_password = os.getenv('redis_password','None')
# print(redis_password)
redis_client = Redis(host=redis_host, port=redis_port ,db=redis_db,decode_responses=True)
# redis_client.flushdb()
# TODO: Add explict list of allowed origins

app = FastAPI(title = 'MultiModal Chatbot')


# TODO: Add explict list of allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request bodies
class TryItNowAudioToText(BaseModel):
    """  schema to generate the text out of audio file  """
    audio_path: str

    class config:
        schema_extra = {
          "example":{
           'audio_path':'/home/mboyina/Music/speech.wav'
           }
           }

class Configuration(BaseModel):
    """ schema to set the configurations and build the chatbot with the configurations requested 
    """
    reload:bool=True
    version: str
    history_window:int
    input_text_id:str
    embedding_model_name: str
    inference_model_name: str
    vector_db_path: str
    search_threshold: float=0.3
    persona: str
    instructions: str
    top_docs: int=5 
    temp: float=0.7
    context_window: int=2048
    repetition_penalty: float=1.1
    top_k: int=40
    top_p: float=0.9
    history:str 
    input_text:str
    webhook:str
    class config:
        schema_extra = {
          "example":{
            "reload":"true",
            "embedding_model_name": "bgelarge",
            "vector_db_path": "/home/mboyina/Music/66866e9559aec1c5d6021a26/66866ebf59aec1c5d6021a29/mm/collection1/S9WSWCNA6M",
            "inference_model_name": "llama3",
            "persona": "You are a friendly and helpful assistant, always ready to assist with any questions and Q/A tasks. Your goal is to provide clear and accurate answers, making sure the information you share is useful and easy to understand. With a positive attitude and a dedication to helping others, you strive to make every interaction smooth and efficient.",
            "search_threshold": 0.1,
            "top_docs": 3.0,
            "temp": 0.7, "context_window": 2048.0, 
            "repetition_penality": 1.1, 
            "top_k": 40.0,
            "top_p": 0.9, 
            "input_text_id": "66b0731676e96e08a09505ba", 
            "input_text": "hi", 
            "history": "{\"/home/mboyina/Music/66866e9559aec1c5d6021a26/66866ebf59aec1c5d6021a29/mm/collection1/EUUDN7VDSP\": {\"type\": \"InMemoryChatMessageHistory\", \"messages\": [{\"type\": \"HumanMessage\", \"content\": \" What is the tank capacity of the car?\"}, {\"type\": \"AIMessage\", \"content\": \"I'm happy to help! Unfortunately, the provided technical specifications do not include information about the tank capacity of the URUS S. However, I can suggest checking the manufacturer's documentation or contacting a dealership for more detailed information on this topic. If you have any other questions, feel free to ask!\"}]}, \"/home/mboyina/Music/66866e9559aec1c5d6021a26/66866ebf59aec1c5d6021a29/mm/collection1/X9BDMSTIWL\": {\"type\": \"InMemoryChatMessageHistory\", \"messages\": [{\"type\": \"HumanMessage\", \"content\": \" What is the document about?\"}, {\"type\": \"AIMessage\", \"content\": \"I'm happy to help!\\n\\nIt looks like we have a digital brochure here! It appears to be a digital publication that provides information and details about something. Without more context, it's hard to say exactly what the brochure is about, but I can take an educated guess.\\n\\nBased on the repeated title \\\"DIGITAL BROCHURE 2\\\", I'm assuming this might be some kind of promotional material or an informational guide for a product, service, or organization. Perhaps it's highlighting features, benefits, and testimonials? Or maybe it's showcasing a portfolio or showcase reel?\\n\\nIf you'd like to share more context or details about the brochure, I'd be happy to help you understand what it's about!\"}]}}",
            "history_window":6,
            "version":'version_1',
            "instructions": "1. Utilize the retrieved information to answer the question. 2. If no information is provided, answer based on your knowledge. 3.Ensure your responses are socially unbiased and positive in nature. Do not include any harmful, unethical, racist, sexist, toxic, dangerous, abusive, or illegal content.",
            "webhook": "http://localhost:9000/multimodel_chatbot/conversations/webhooks/questions/66b0731676e96e08a09505ba/answer/"}
           }



class HandleText(BaseModel):
    
    """ schema which accepts the input text and generated the answer. """
    
    input_text: str
    input_id: str
    webhook: str

    class config:
        schema_extra = {
            "example" : {
            "input_text":"what does the document talk about",
            "input_id": "66b0731676e96e08a09505ba",
            "webhook": "http://localhost:9000/multimodel_chatbot/conversations/webhooks/questions/66b0731676e96e08a09505ba/answer/"
            }
        }


class Customized_Rag_Vectordb_Creation(BaseModel):

    """ schema for creating the vector database """

    documents_path :str 
    embedding_model_name:str
    vectorindex_name:str
    chunk_size:int
    chunk_overlap:int

    class config:
        schema_extra = {
            "example":{
                "documents_path" : "/home/user/cardata",
                "embedding_model_name":"bgelarge",
                "vectorindex_name":"car_vectord_db",
                "chunk_size":"1200",
                "chunk_overlap":"200"
            }
        }

def push_data_into_redis_list(Configuration):
    
    """
    This function inserts data into a Redis client, which will later be used to check the status and compare configurations.
    If the configurations are the same as the previous request, the chatbot will not be rebuilt; instead, 
    the pre-built chatbot will be used to generate an answer.
    """
    config_dict =Configuration.dict()
    redis_client.rpush(config_dict['vector_db_path'],json.dumps(config_dict))
    return config_dict


def last_pushed_configiration_data(Configuration):
    
    """ 
    This function is to get last configurations to compare with current configurations.
    """
    config_dict =Configuration.dict()
    previous_request = redis_client.lindex(config_dict['vector_db_path'],-1)
    
    if previous_request==None:
        return {}
    else:
        previous_request = json.loads(previous_request)
        return previous_request

def bot_creation_status_check(config: Configuration):

    """
    Functions to process questions asynchronously .
    1.pushes data into redis
    2.gets previous data
    3.compares both configurations
    4.either creates chatbot if configurations are not same or gets the answer from already built chatbot
    5.updates the redis client with status
    """

    redis_client.hset(config. input_text_id, mapping={'status': 'In Progress', 'result': ''}) 

    last_pushed_config =last_pushed_configiration_data(config)
    current_config = push_data_into_redis_list(config)

    if last_pushed_config == {}:
        same_configiration=False
    else:
       same_configiration = (last_pushed_config['version'] == current_config['version'])
       
    if same_configiration and config.reload==False:
        
        try:
            result = Tasks().get_answer( config.embedding_model_name,
                config.inference_model_name,
                config.vector_db_path,
                config.search_threshold,
                config.persona,
                config.instructions,
                config.top_docs,
                config.temp,
                config.context_window,
                config.repetition_penalty,
                config.top_k,
                config.top_p,
                config.history,
                config.input_text,
                config.version,
                config.history_window,
                config.webhook,)
            redis_client.hset(config.input_text_id, mapping={'status': 'Completed', 'result': str(result)})
        except:
            redis_client.hset(config.input_text_id, mapping={'status': 'Failed', 'result': f"Failed: {str(e)}"})   
            print(f"Failed:{e}") 
    else:    
        try:

            result = Tasks().bot_creation_with_configiration(
                config.embedding_model_name,
                config.inference_model_name,
                config.vector_db_path,
                config.search_threshold,
                config.persona,
                config.instructions,
                config.top_docs,
                config.temp,
                config.context_window,
                config.repetition_penalty,
                config.top_k,
                config.top_p,
                config.history,
                config.input_text,
                config.version,
                config.history_window,
                config.webhook,
            )
            redis_client.hset(config.input_text_id, mapping={'status': 'Completed', 'result': str(result)})
        except Exception as e:
            print(f"Failed due to :{e}")
            redis_client.hset(config.input_text_id, mapping={'status': 'Failed', 'result': f"Failed: {str(e)}"})


@app.post("/Customized_Bot/audio_to_text",summary="converts the audio to text")
async def audio_to_text(request: TryItNowAudioToText):
    """
    Converts the audio to text 
    \n **Accepts**\n
    'audio_path':'/home/mboyina/Music/speech.wav'
    \n **Returns**\n
    response_text:"............."

    """
    try:
        recognized_text = transcribe_audio_from_file(request.audio_path)
        return {'response_text': recognized_text}
    except Exception as e:
        return {'Error': f'Failed: {str(e)}'}


@app.post( "/Customized_Bot/bot_creation", summary="creates the customized chatbot and answers the question" )
async def bot_creation(request: Configuration, background_tasks: BackgroundTasks):
    """
    Creates the chatbot with user configurations and adds the request details to redis client &
    Answers a question
   \n
    **Accepts**\n
    all configurations \n
    **Returns**\n
    'message': 'Successfully started creating custom bot',
    'request_id': request.input_text_id

    """

    if redis_client.exists(request.input_text_id):
        raise HTTPException(status_code=400, detail="Question ID already exists")
    background_tasks.add_task(bot_creation_status_check, request)
    sleep(0.5)
    redis_client.hset(request.input_text_id, mapping={'status': 'Pending', 'result': ''})
    
    return {'message': 'Successfully started creating custom bot', 'request_id': request.input_text_id}


# Endpoint to get status of the question
@app.get('/Customized_Bot/get_status',summary="get status of the task")
async def get_status(input_id: str):
    """
    gets status of the task by input_id.\n
    **Accepts**:\n
    input_id:898ehhddf9gfgewf9
    \n
    **Returns**\n
    status:completed \n
    result:{responsetext:''}/{answer:}
    """

    if not redis_client.exists(input_id):
        raise HTTPException(status_code=404, detail="Question ID not found")

    task_info = redis_client.hgetall(input_id)
    return task_info


@app.post('/Customized_Bot/embedidng_creation',summary="Creates vectordata base.")
async def create_vectord_db(request:Customized_Rag_Vectordb_Creation ,backgroundtasks:BackgroundTasks):
  """
    Create the embeddings and store them in chroma vectorstore.\n
    **Accepts**\n
    "documents_path" : "/home/user/cardata",
    "embedding_model_name":"bgelarge",
    "vectorindex_name":"car_vectord_db",
    "chunk_size":"1200",
    "chunk_overlap":"200"
    \n**Returns**\n
    'message':'succesfully started creating vectordb',\n
    'vectordbpath':vector_db_path}

  """


  res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=10))

  vector_db_path = request.documents_path+"/"+res
#   url = os.getenv("plot_url")
#   plot_name="/home/mboyina/Music/testingragdata/gitaccesstoken_docs.docx"
#   plot_url = url.format(plot_path =plot_name)
  backgroundtasks.add_task(Tasks().create_customized_vb,request.documents_path,vector_db_path,request.embedding_model_name,request.chunk_size,request.chunk_overlap)
  sleep(0.5)
  return{'message':'succesfully started creating vectordb','vectordbpath':vector_db_path}

