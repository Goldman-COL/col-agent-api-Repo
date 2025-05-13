import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import time
import uuid

load_dotenv()

from fastapi import FastAPI app = FastAPI() @app.get("/health") async def health():     return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info",
                access_log=True, use_colors=True)
