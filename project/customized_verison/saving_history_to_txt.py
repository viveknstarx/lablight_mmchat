import json
from typing import Dict, Any
from langchain.schema import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (HumanMessage, AIMessage)):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content
            }
        elif isinstance(obj, InMemoryChatMessageHistory):
            return {
                "type": "InMemoryChatMessageHistory",
                "messages": obj.messages
            }
        return super().default(obj)

