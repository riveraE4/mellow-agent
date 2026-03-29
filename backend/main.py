import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

load_dotenv()
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    job_posting: str
    user_profile: dict


@app.get("/")
def root():
    return {"status": "Mellow API is running"}


@app.post("/match_job")
async def match_job(req: MatchRequest):
    prompt = f"""
You are a job matching assistant.

User profile:
{json.dumps(req.user_profile, indent=2)}

Job posting:
{req.job_posting}

Return ONLY valid JSON:
{{
  "matchScore": <0-100>,
  "reasoning": "<2-3 sentences>",
  "recommended": <true|false>,
  "missingSkills": ["<skill>"],
  "highlights": ["<reason>"]
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
