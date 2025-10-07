from pydantic import BaseModel

class FloorplanResponse(BaseModel):
    status: str
    image_url: str
    result: dict
