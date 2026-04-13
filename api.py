import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Load env variables FIRST
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            if _key.strip() not in os.environ:
                os.environ[_key.strip()] = _val.strip().strip('"').strip("'")

# Now import RAG system
from rag_system import RAGSystem

app = FastAPI(title="Advanced RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize single instance of RAG System lazily or globally
_rag_system = None

def get_rag():
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system

class QueryRequest(BaseModel):
    question: str
    mode: str = "auto"

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    try:
        rag = get_rag()
        if req.mode == "direct":
            res = rag.query(req.question, verify=False, verbose=True)
            res["mode"] = "direct"
        elif req.mode == "agent":
            res = rag.agent_query(req.question)
            res["mode"] = "agent"
        else: # auto
            res = rag.auto_query(req.question, verify=False, verbose=True)
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ToolRequest(BaseModel):
    name: str
    input: dict = {}

@app.get("/api/tools")
async def tools_info_endpoint():
    try:
        rag = get_rag()
        return {"tools": rag.list_tools()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tool")
async def tool_run_endpoint(req: ToolRequest):
    try:
        rag = get_rag()
        result = rag.run_tool(req.name, req.input)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def stats_endpoint():
    try:
        rag = get_rag()
        return rag.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    tmp_path = None
    try:
        rag = get_rag()
        suffix = Path(file.filename).suffix if file.filename else ".txt"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        result = rag.ingest(tmp_path)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Ensure frontend directory exists
Path("frontend").mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
