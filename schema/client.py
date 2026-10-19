from pydantic import BaseModel
from typing import Dict

class ToolModel(BaseModel):
    name: str
    description: str
    parameters: Dict[str,object]