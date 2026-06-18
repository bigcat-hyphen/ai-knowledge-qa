from pydantic import BaseModel, Field
from typing import Optional


class ConfigSchema(BaseModel):
    api_key: str = Field(..., min_length=1, max_length=4096)
    base_url: str = Field(..., min_length=1, max_length=4096)
    model: str = Field(..., min_length=1, max_length=256)
    embed_model: Optional[str] = Field(default=None, max_length=256)


class AskSchema(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    chat_history: list[dict] = Field(default_factory=list)


class ConversationSchema(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    messages: list[dict] = Field(..., min_length=1)
