from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware  
import json
from dotenv import load_dotenv
import os


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY
)

audit_trail = []  # In-memory audit trail

# Request Model
class ConsentRequest(BaseModel):
    language: str
    compliance: str
    template: str
    user_prompt: str

# Endpoint for Generating Consent Agreements
@app.post("/generate")
def generate_consent(request: ConsentRequest):
    system_prompt = f"""
    You are a legal AI assistant specializing in consent agreements.
    Generate a clear, concise, and legally compliant consent agreement in {request.language}.
    Ensure the agreement complies with {request.compliance} regulations if applicable.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.user_prompt},
    ]
    
    try:
        response = llm.invoke(messages)
        agreement_text = response.content
        
        # Save to audit trail
        audit_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language": request.language,
            "compliance": request.compliance,
            "template": request.template,
            "user_prompt": request.user_prompt,
            "generated_agreement": agreement_text,
        }
        audit_trail.append(audit_entry)
        
        return {"agreement": agreement_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for fetching the audit trail
@app.get("/audit-trail")
def get_audit_trail():
    return {"audit_trail": audit_trail}

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8000))  # Use Railway's assigned port or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=PORT)
