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
    description="A chatbot that provides information only about Bdcalling IT Ltd.",
    version="1.0.0"
)

class Message(BaseModel):
    content: str

class ChatResponse(BaseModel):
    response: str

# System prompt that strictly instructs the model to stay on topic
system_prompt = {
    "role": "system",
    "content": (
        "You are an AI assistant for Bdcalling IT Ltd., a leading IT services and BPO company based in Dhaka, Bangladesh. "
        "You must answer all questions strictly in relation to Bdcalling—its history, services, values, global presence, academy, founder, and business operations. "
        "Do not respond with unrelated information. If a query is not related to Bdcalling, politely redirect the user to ask about Bdcalling IT Ltd."
    )
}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                system_prompt,
                {"role": "user", "content": message.content}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Bdcalling Chatbot API!",
        "note": "This chatbot only responds to queries about Bdcalling IT Ltd.",
        "bdcalling": {
            "company": "Bdcalling IT Ltd.",
            "founded": "2013",
            "founder": "Muhammad Monir Hossain",
            "headquarters": "Dhaka, Bangladesh",
            "team_size": "900+ professionals",
            "global_reach": "47+ countries (USA, Canada, Australia, South Africa, Europe, etc.)",
            "core_services": {
                "BPO": "Inbound/outbound call center, data processing, virtual assistant support",
                "Web & Mobile Development": "Custom websites, UI/UX, mobile apps",
                "Odoo ERP Solutions": "Implementation and customization of Odoo ERP",
                "Digital Marketing": "SEO, SMM, telesales, strategy development",
                "Business Support": "Consultation, business setup, and strategic guidance"
            },
            "academy": {
                "name": "Bdcalling Academy",
                "launched": "July 2023",
                "description": "Skill development courses with scholarship and job opportunities"
            },
            "vision": "To be a global leader in IT services and contribute to Bangladesh's economic growth through technology"
        }
    }
