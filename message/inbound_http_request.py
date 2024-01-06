from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str
    embed_model : str