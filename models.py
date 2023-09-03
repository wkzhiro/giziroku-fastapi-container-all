from pydantic import BaseModel

class MeetingData(BaseModel):
    title: str
    date: str
    participants: list[str]
    purpose: str
    precision:str

class json_file(BaseModel):
    filename : str