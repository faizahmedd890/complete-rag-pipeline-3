from fastapi import APIRouter

from app.schemas.auth import (
    UserRequest,
    UserResponse
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post(
    "/verify",
    response_model=UserResponse
)
async def verify_user(
    request: UserRequest
):

    return UserResponse(
        user_id=request.user_id,
        authenticated=True
    )