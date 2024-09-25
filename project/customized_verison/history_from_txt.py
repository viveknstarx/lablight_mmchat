import json
from typing import Dict, Union
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

def load_chat_history_from_file(filename: str) -> Union[Dict[str, InMemoryChatMessageHistory], Dict]:
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        if not content.strip():  # Check if the file is empty or contains only whitespace
            return {}
        data = json.loads(content)
    if not data:  # If the dictionary is empty
        return {}
    
    reconstructed_history = {}
    for key, value in data.items():
        if value['type'] == 'InMemoryChatMessageHistory':
            messages = []
            for msg in value['messages']:
                if msg['type'] == 'HumanMessage':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['type'] == 'AIMessage':
                    messages.append(AIMessage(content=msg['content']))
            reconstructed_history[key] = InMemoryChatMessageHistory(messages=messages)
    
    return reconstructed_history

# a = load_chat_history_from_file("/home/mboyina/Music/SHI/MM_customized _version/MM_CUSTOMIZED/project/Main/chat_history.txt")
# print(a)

# data="{\"/home/mboyina/Music/embeding data/luxury_cars-20240717T045912Z-001/luxury_cars\": {\"type\": \"InMemoryChatMessageHistory\", \"messages\": [{\"type\": \"HumanMessage\", \"content\": \"hi\"}, {\"type\": \"AIMessage\", \"content\": \"(big smile) Ah, hello there! Welcome to our dealership! I'm Bob, one of the sales managers here. What can I do for you today? Are you looking for a new ride or just browsing? We've got some fantastic deals going on right now, and I'd be happy to show you around.\"}, {\"type\": \"HumanMessage\", \"content\": \"what is my name?\"}, {\"type\": \"AIMessage\", \"content\": \"Good question! Let me check that for ya. (checks computer) Ah yes, according to our system, your name is... (dramatic pause) ...John! That's right, John. And how can I help you today, John? You looking for a new car or just taking a look around? We've got some amazing deals on some fantastic vehicles that I think you'd really like. What do you say we take a walk and see what catches your eye?\"}, {\"type\": \"HumanMessage\", \"content\": \"what is my name?\"}, {\"type\": \"AIMessage\", \"content\": \"(big smile) Ah, good question! Let me check that for ya. (checks computer) Ah yes, according to our system, your name is... (dramatic pause) ...John! That's right, John. And how can I help you today, John? You looking for a new car or just taking a look around? We've got some amazing deals on some fantastic vehicles that I think you'd really like. What do you say we take a walk and see what catches your eye?\"}, {\"type\": \"HumanMessage\", \"content\": \"hi\"}, {\"type\": \"AIMessage\", \"content\": \"(big smile) Ah, hello there again! Welcome back to our dealership! I'm Bob, one of the sales managers here. How can I help you today? You were just asking about your name, right? (checks computer) Let me check that for ya... (dramatic pause) ...Ah yes, according to our system, your name is still... (drumroll) ...John! That's right, John. And how can I help you today, John? You looking for a new car or just taking a look around? We've got some fantastic deals going on right now, and I'd be happy to show you around. What do you say we take a walk and see what catches your eye?\"}]}}"
# data=json.loads(data)
# reconstructed_history = {}
# for key, value in data.items():
#         if value['type'] == 'InMemoryChatMessageHistory':
#             messages = []
#             for msg in value['messages']:
#                 if msg['type'] == 'HumanMessage':
#                     messages.append(HumanMessage(content=msg['content']))
#                 elif msg['type'] == 'AIMessage':
#                     messages.append(AIMessage(content=msg['content']))
#             reconstructed_history[key] = InMemoryChatMessageHistory(messages=messages)
    
# print(reconstructed_history)

def format_history(data):
    data=json.loads(data)
    reconstructed_history = {}
    for key, value in data.items():
        if value['type'] == 'InMemoryChatMessageHistory':
            messages = []
            for msg in value['messages']:
                if msg['type'] == 'HumanMessage':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['type'] == 'AIMessage':
                    messages.append(AIMessage(content=msg['content']))
            reconstructed_history[key] = InMemoryChatMessageHistory(messages=messages)
    return reconstructed_history
