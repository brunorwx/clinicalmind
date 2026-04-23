# app/auth.py
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel
from app.config import settings

bearer = HTTPBearer()

class CurrentUser(BaseModel):
    id: str
    email: str
    trial_access: list[str]

def create_token(user_id: str, email: str, trial_access: list[str]) -> str:
    payload = {
        "sub": user_id, "email": email, "trials": trial_access,
        "exp": datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer)
) -> CurrentUser:
    try:
        payload = jwt.decode(creds.credentials, settings.jwt_secret,
                             algorithms=[settings.jwt_algorithm])
        return CurrentUser(
            id=payload["sub"],
            email=payload["email"],
            trial_access=payload.get("trials", []),
        )
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token")