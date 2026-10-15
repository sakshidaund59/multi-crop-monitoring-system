import re
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
VALID_ROLES = {"farmer", "agronomist", "admin"}


class UserRegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6, max_length=72, description="Password between 6 and 72 characters")
    full_name: str = Field(..., min_length=1, max_length=100)
    role: str = Field(default="farmer")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Email must be a string")
        clean_email = v.strip().lower()
        if not EMAIL_REGEX.match(clean_email):
            raise ValueError("Invalid email format")
        return clean_email

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Role must be a string")
        clean_role = v.strip().lower()
        if "farmer" in clean_role:
            return "farmer"
        if "agronomist" in clean_role:
            return "agronomist"
        if "admin" in clean_role:
            return "admin"
        if clean_role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{v}'. Allowed roles: farmer, agronomist, admin")
        return clean_role


class UserLoginRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=1, max_length=72)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Email must be a string")
        clean_email = v.strip().lower()
        if not EMAIL_REGEX.match(clean_email):
            raise ValueError("Invalid email format")
        return clean_email


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
