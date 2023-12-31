from pydantic import BaseModel 

class ChatResponse(BaseModel):
    text: str
