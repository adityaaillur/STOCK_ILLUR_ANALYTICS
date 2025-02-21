from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from modules.database.manage_users import (
    create_user, 
    update_user,
    delete_user,
    reset_password
)
from modules.middleware.auth import (
    get_current_active_user,
    protect,
    Permission,
    create_password_reset_token,
    verify_password_reset_token
)
from modules.models.user import User
from modules.schemas import UserCreate, UserUpdate, PasswordReset
from modules.config import settings
from services.email import EmailService

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", dependencies=[Depends(protect([Permission.MANAGE_USERS]))])
async def create_new_user(user: UserCreate):
    return await create_user(
        username=user.username,
        email=user.email,
        password=user.password,
        roles=user.roles
    )

@router.put("/{username}", dependencies=[Depends(protect([Permission.MANAGE_USERS]))])
async def update_existing_user(username: str, user: UserUpdate):
    return await update_user(
        username=username,
        new_email=user.email,
        new_roles=user.roles,
        is_active=user.is_active,
        is_superuser=user.is_superuser
    )

@router.delete("/{username}", dependencies=[Depends(protect([Permission.MANAGE_USERS]))])
async def delete_existing_user(username: str):
    success = await delete_user(username)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}

@router.post("/password-reset-request")
async def request_password_reset(email: str):
    reset_token = create_password_reset_token(email)
    reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
    
    # Send email
    email_service = EmailService()
    success = await email_service.send_password_reset_email(email, reset_link)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to send reset email"
        )
    
    return {"message": "Password reset link sent to email"}

@router.post("/reset-password")
async def complete_password_reset(reset_data: PasswordReset):
    email = await verify_password_reset_token(reset_data.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )
    
    success = await reset_password(email, reset_data.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password reset failed"
        )
    
    return {"message": "Password updated successfully"} 