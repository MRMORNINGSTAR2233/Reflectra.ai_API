
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId
from db import db

# FastAPI app instance
app = FastAPI()