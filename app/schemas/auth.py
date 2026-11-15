from pydantic import BaseModel, Field


class UserRequest(BaseModel):

    user_id: str = Field(
        min_length=1,
        max_length=200
    )


class UserResponse(BaseModel):

    user_id: str

    authenticated: bool