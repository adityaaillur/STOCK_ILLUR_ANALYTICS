from fastapi import HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pydantic import BaseModel
from ..config import settings
from database.session import AsyncSessionLocal
from models.activity import ActivityLog

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    username: Optional[str] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data

class Permission:
    VIEW_DASHBOARD = "view:dashboard"
    EXECUTE_TRADES = "execute:trades"
    MANAGE_USERS = "manage:users"
    CONFIGURE_SYSTEM = "configure:system"
    VIEW_ANALYTICS = "view:analytics"

class Role:
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    GUEST = "guest"

role_permissions: Dict[str, List[str]] = {
    Role.ADMIN: [Permission.VIEW_DASHBOARD, Permission.EXECUTE_TRADES,
                 Permission.MANAGE_USERS, Permission.CONFIGURE_SYSTEM],
    Role.TRADER: [Permission.VIEW_DASHBOARD, Permission.EXECUTE_TRADES],
    Role.ANALYST: [Permission.VIEW_DASHBOARD, Permission.VIEW_ANALYTICS],
    Role.GUEST: [Permission.VIEW_DASHBOARD]
}

class User(BaseModel):
    user_id: str
    username: str
    email: str
    roles: List[str] = [Role.GUEST]
    disabled: bool = False

class RBACMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, request: Request, call_next):
        # Get current user from request state
        user: Optional[User] = request.state.user
        
        # Get required permissions for the endpoint
        required_perms = request.scope.get("required_permissions", [])
        
        if not user or not self.has_permission(user, required_perms):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to access this resource"
            )
            
        return await call_next(request)
    
    def has_permission(self, user: User, required_perms: List[str]) -> bool:
        if not required_perms:
            return True
            
        user_perms = set()
        for role in user.roles:
            user_perms.update(role_permissions.get(role, []))
            
        return all(perm in user_perms for perm in required_perms)

def protect(endpoint, required_permissions: List[str] = []):
    def wrapper(*args, **kwargs):
        return endpoint(*args, **kwargs)
    wrapper.required_permissions = required_permissions
    return wrapper 

class ActivityType:
    LOGIN = "user:login"
    TRADE = "trade:execute"
    CONFIG_CHANGE = "config:update"
    USER_MOD = "user:modify"
    DATA_ACCESS = "data:access"

class ActivityService:
    async def log_activity(self, 
                          user_id: str,
                          activity_type: str,
                          endpoint: str,
                          metadata: dict = {}):
        """Log user activity to database"""
        async with AsyncSessionLocal() as session:
            activity = ActivityLog(
                user_id=user_id,
                activity_type=activity_type,
                endpoint=endpoint,
                metadata=metadata,
                timestamp=datetime.utcnow()
            )
            session.add(activity)
            await session.commit()

class AuditMiddleware:
    def __init__(self, app):
        self.app = app
        self.activity_svc = ActivityService()
        
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        
        # Get user from request state
        user = request.state.user
        
        if user and not user.disabled:
            await self.activity_svc.log_activity(
                user_id=user.user_id,
                activity_type=self._get_activity_type(request),
                endpoint=request.url.path,
                metadata={
                    "method": request.method,
                    "status_code": response.status_code,
                    "params": dict(request.query_params),
                    "client": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                }
            )
            
        return response
    
    def _get_activity_type(self, request: Request) -> str:
        if request.method == "POST" and "/trades" in request.url.path:
            return ActivityType.TRADE
        elif request.method == "PUT" and "/users" in request.url.path:
            return ActivityType.USER_MOD
        elif request.method == "PATCH" and "/config" in request.url.path:
            return ActivityType.CONFIG_CHANGE
        return ActivityType.DATA_ACCESS

# Add to existing ActivityLog model (in database models)
"""
class ActivityLog(Base):
    __tablename__ = "activity_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    activity_type = Column(String)
    endpoint = Column(String)
    metadata = Column(JSON)
    timestamp = Column(DateTime)
""" 

# Add to Settings class in config.py
PASSWORD_RESET_SECRET: str = "your-reset-secret-key"
RESET_TOKEN_EXPIRE_MINUTES: int = 30

def create_password_reset_token(email: str) -> str:
    expires = datetime.utcnow() + timedelta(minutes=settings.RESET_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": email, "exp": expires}
    return jwt.encode(to_encode, settings.PASSWORD_RESET_SECRET, algorithm=settings.ALGORITHM)

async def verify_password_reset_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.PASSWORD_RESET_SECRET, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None 