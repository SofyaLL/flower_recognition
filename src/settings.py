from pydantic import Field
from pydantic_settings import BaseSettings


class UvicornSettings(BaseSettings):
    host: str = Field(env="UVICORN_HOST", default="0.0.0.0")
    port: int = Field(env="UVICORN_PORT", default=8100)
