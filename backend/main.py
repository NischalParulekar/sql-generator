from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp_sql import generate_sql

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    prompt: str
    schema: str = ""


@app.post("/generate-sql")
def generate(request: QueryRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    sql = generate_sql(request.prompt, request.schema)
    return {"sql": sql}
