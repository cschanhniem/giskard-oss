from pydantic import BaseModel

from .chat import CompletionUsage

class ResponseParameters(BaseModel, extra="ignore"):
    