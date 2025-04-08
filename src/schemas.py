from pydantic import BaseModel

# Define input data model
class Tweet(BaseModel):
    text: str

