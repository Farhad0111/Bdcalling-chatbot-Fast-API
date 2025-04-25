from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Check if API key exists
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Initialize Groq client
client = groq.Groq(
    api_key=api_key
)

# FastAPI app
app = FastAPI(
    title="Bdcalling Chatbot",
    description="A simple chatbot API using FastAPI and Groq",
    version="1.0.0"
)

class Message(BaseModel):
    content: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Groq's recommended model
            messages=[
                {"role": "user", "content": message.content}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Bdcalling Chatbot API!"}

