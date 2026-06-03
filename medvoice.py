
import os
import re
import uuid
import datetime
import chromadb
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
import anyio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_text_splitters import RecursiveCharacterTextSplitter
from faster_whisper import WhisperModel

# --- ML & SYSTEM CORE FRAMEWORKS ---
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess
# Document Extractors
from pypdf import PdfReader
import docx
import uuid
import os
import torchaudio
from styletts2 import tts
import soundfile as sf
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize
# --- REAL DATABASE INFRASTRUCTURE (SQLAlchemy + SQLite) ---
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Date, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

#torchaudio.set_audio_backend("soundfile")
DATABASE_URL = "sqlite:///./healthcare_agent.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- SQLALCHEMY DATABASE MODELS ---
class Patient(Base):
    __tablename__ = "patients"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    sex = Column(String)
    condition = Column(String)
    insurance_plan = Column(String)
    
    visits = relationship("Visit", back_populates="patient", cascade="all, delete-orphan")
    medications = relationship("Medication", back_populates="patient", cascade="all, delete-orphan")
    allergies = relationship("Allergy", back_populates="patient", cascade="all, delete-orphan")

class Visit(Base):
    __tablename__ = "visits"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    visit_date = Column(Date, default=datetime.date.today)
    diagnosis = Column(String)
    doctor_notes = Column(Text)
    
    patient = relationship("Patient", back_populates="visits")

class Medication(Base):
    __tablename__ = "medications"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    medication_name = Column(String, nullable=False)
    dosage = Column(String)
    frequency = Column(String)
    
    patient = relationship("Patient", back_populates="medications")

class Allergy(Base):
    __tablename__ = "allergies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    allergy_name = Column(String, nullable=False)
    severity = Column(String)
    
    patient = relationship("Patient", back_populates="allergies")

# Create tables and seed data instantly on application start
Base.metadata.create_all(bind=engine)

def seed_database_if_empty():
    db = SessionLocal()
    try:
        if db.query(Patient).count() == 0:
            p1 = Patient(id="P001", name="Sri Sai", age=25, sex="Female", condition="Hypertension", insurance_plan="Tier-1 Premium")
            p2 = Patient(id="P002", name="John Doe", age=45, sex="Male", condition="Diabetes Type II", insurance_plan="Standard Basic")
            
            db.add_all([p1, p2])
            db.commit()
            
            v1 = Visit(patient_id="P001", visit_date=datetime.date(2026, 5, 15), diagnosis="Routine follow-up", doctor_notes="Blood pressure fluctuating slightly. Rest advised.")
            m1 = Medication(patient_id="P001", medication_name="Metformin", dosage="500mg", frequency="Once daily")
            a1 = Allergy(patient_id="P002", allergy_name="Penicillin", severity="High")
            
            db.add_all([v1, m1, a1])
            db.commit()
            logging.info("SQLite Database successfully initialized and seeded with structural parameters.")
    except Exception as e:
        db.rollback()
        logging.error(f"Seeding failure: {str(e)}")
    finally:
        db.close()

seed_database_if_empty()

# --- APP CONFIGURATION & SECURITY SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MultiModalAgent")

app = FastAPI(
    title="Autonomous Healthcare Ops & Infrastructure Agent",
    description="Multimodal Local Agent handling Voice (STT/TTS), Advanced RAG (FAISS), and Real SQLite Database Layers.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TTS_DIR = "tts_cache"
os.makedirs(TTS_DIR, exist_ok=True)
UPLOAD_DIR = "uploaded_docs"
AUDIO_DIR = "audio_cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")
app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Loading Whisper model...")

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

logger.info(
    "Whisper initialized."
)

# --- GLOBAL MODEL INITIALIZATION ---
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Mistral-7B-Instruct-v03-IQ4_NL.gguf")

# --- GLOBAL MODEL INITIALIZATION ---
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Mistral-7B-Instruct-v03-IQ4_NL.gguf")

try:
    llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=2048, n_threads=6, verbose=False)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize high-quality StyleTTS 2 framework directly in RAM
    logger.info("Loading Pretrained StyleTTS 2 Engine into memory...")
    
    # SYSTEM-LEVEL OVERRIDE FOR PYTORCH 2.6+ TO RESTORE LEGACY UNPICKLING
    import torch
    import functools
    
    # Save original torch.load function reference
    original_torch_load = torch.load
    
    # Force weights_only=False globally for any underlying packages calling it
    torch.load = functools.partial(original_torch_load, weights_only=False)
    
    try:
        styletts_engine = tts.StyleTTS2()
    finally:
        # Instantly restore clean PyTorch security standards right after loading
        torch.load = original_torch_load
        
    print("working")
except Exception as e:
    logger.error(f"Failed to load local foundational models: {str(e)}")
    raise RuntimeError("Ensure model files are placed in correct paths.")

# --- CHROMADB PERSISTENT VECTOR STORE ---
chroma_client = chromadb.PersistentClient(
    path="./vector_db"
)

collection = chroma_client.get_or_create_collection(
    name="medical_knowledge"
)

# --- PYDANTIC SCHEMAS ---
class TextChatRequest(BaseModel):
    text_query: str = Field(..., example="What are the doctor notes for Sri Sai?")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

# --- CORE UTILITY CONTROLLERS & SERVICES ---
def sanitize_input(text: str) -> str:
    if not text: return ""
    clean = re.sub(r"<script.*?>.*?</script.*?>", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"[<>\'\"\\;]", "", clean)
    return clean.strip()

def generate_f5_tts(text: str) -> str:
    if not text.strip():
        return None

    clean_text = text.replace("\n", " ").replace("*", " ").replace("`", " ")
    clean_text = clean_text.replace("|", ", ").replace("---", " ")
    clean_text = re.sub(r'\berpdf\b', 'e-p-d-f', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\bpdf\b', 'p-d-f', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(TTS_DIR, output_filename)

    try:
        os.makedirs(TTS_DIR, exist_ok=True)
        # Tokenize sentences
        all_sentences = sent_tokenize(clean_text)
        
        # --- SPEED PATCH: Only process the first 2 sentences for instant playback ---
        sentences = all_sentences[:2] 
        logger.info(f"Speed Optimization Active: Processing top {len(sentences)} chunks.")

        combined_audio = []
        sample_rate = 24000

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # --- SPEED PATCH: Dropped diffusion_steps to 3 ---
            audio_chunk = styletts_engine.inference(
                text=sentence,
                target_voice_path="voice_reference/hayagriva_ref.wav",
                diffusion_steps=3,  
                alpha=0.3,
                beta=0.7,
                output_sample_rate=sample_rate
            )
            
            if hasattr(audio_chunk, "cpu"):
                audio_chunk = audio_chunk.cpu().numpy()
            audio_chunk = np.squeeze(audio_chunk)
            
            combined_audio.append(audio_chunk)
            silence_buffer = np.zeros(int(0.15 * sample_rate), dtype=audio_chunk.dtype)
            combined_audio.append(silence_buffer)

        if not combined_audio:
            return None

        final_audio_arr = np.concatenate(combined_audio, axis=0)
        sf.write(output_path, final_audio_arr, sample_rate)
        return output_filename
    except Exception as e:
        logger.error(f"StyleTTS 2 optimized generation failed: {str(e)}")
        return None
def transcribe_audio(audio_path):

    segments, info = whisper_model.transcribe(
        audio_path,
        beam_size=1
    )

    transcript = " ".join(
        [segment.text for segment in segments]
    )

    return transcript.strip()

def run_orm_database_query(user_query: str) -> str:
    """Real Database Service: Parses keywords to retrieve context via SQLAlchemy ORM"""
    db = SessionLocal()
    query_lower = user_query.lower()
    context_out = ""
    try:
        # Search for explicit user data matching strings
        if "sri sai" in query_lower or "p001" in query_lower:
            patient = db.query(Patient).filter(Patient.name.like("%Sri Sai%")).first()
        elif "john" in query_lower or "p002" in query_lower:
            patient = db.query(Patient).filter(Patient.name.like("%John%")).first()
        else:
            patient = None

        if patient:
            context_out = f"Patient Info: Name={patient.name}, Age={patient.age}, Condition={patient.condition}, Plan={patient.insurance_plan}. "
            if patient.visits:
                context_out += f"Visits: {[f'Date: {v.visit_date}, Notes: {v.doctor_notes}' for v in patient.visits]} "
            if patient.medications:
                context_out += f"Medications: {[f'Name: {m.medication_name}, Dosage: {m.dosage}' for m in patient.medications]} "
            if patient.allergies:
                context_out += f"Allergies: {[f'Name: {a.allergy_name}, Severity: {a.severity}' for a in patient.allergies]} "
        
        # General inventory stock checks
        elif "stock" in query_lower or "inventory" in query_lower or "metformin" in query_lower:
            meds = db.query(Medication).all()
            context_out = f"Active System Medication Registry Logs: {[f'Med: {m.medication_name}, Target Patient ID: {m.patient_id}' for m in meds]}"
            
        else:
            # Fallback total aggregation diagnostic count
            patient_count = db.query(Patient).count()
            visit_count = db.query(Visit).count()
            context_out = f"Database Diagnostic: System connected via SQLAlchemy. Active state context: {patient_count} patient profiles tracking, {visit_count} total historic visit timelines recorded."
            
        return f"Database Service Query Output:\n{context_out}"
    except Exception as e:
        logger.error(f"DB Service layer error: {str(e)}")
        return f"Database Exception encountered during query operation processing."
    finally:
        db.close()

def extract_text_from_file(file_path: str, filename: str) -> str:
    ext = filename.split(".")[-1].lower()
    text = ""
    if ext == "pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
    elif ext == "docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text

def execute_local_inference(prompt: str) -> str:
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are Hayagriva, an advanced enterprise medical operations system. Summarize details professionally in plain text using the retrieved database or context parameters provided. Do not use markdown backticks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=256
    )
    return output["choices"][0]["message"]["content"].strip()

async def process_intent_routing_workflow(user_query: str) -> dict:
    cleaned_query = sanitize_input(user_query)
    
    # Advanced Intent Router logic
    if any(k in cleaned_query.lower() for k in ["stock", "patient", "inventory", "record", "notes", "allergy", "visit", "sri sai", "john"]):
        routing_intent = "SQL"
    elif collection.count() > 0:
        routing_intent = "RAG"
    else:
        routing_intent = "GENERAL_LLM"
        
    context_assembly = ""
    if routing_intent == "SQL":
        # Invoking our dynamic database mapping layer
        context_assembly = run_orm_database_query(cleaned_query)
    elif routing_intent == "RAG":

        query_vector = embedding_model.encode(
            cleaned_query
            
        )
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=3
        )
        retrieved_docs = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):

                retrieved_docs.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "chunk_id": results["metadatas"][0][i]["chunk_id"]
            })
        context_assembly = (
        "Retrieved Medical Knowledge Context:\n"
        + "\n---\n".join([
            f"[Source: {doc['source']} | Chunk: {doc['chunk_id']}]\n{doc['text']}"
            for doc in retrieved_docs
        ])
        )
        
    orchestration_prompt = f"User Request: {cleaned_query}\nRetrieved System Context: {context_assembly if context_assembly else 'No internal context required.'}\nGenerate operational response summary:"
    llm_response = await anyio.to_thread.run_sync(execute_local_inference, orchestration_prompt)
    
    return {
        "intent": routing_intent,
        "confidence": 0.95 if context_assembly else 0.80,
        "response": llm_response
    }

# --- SERVE FRONTEND INDEX ---
@app.get("/", tags=["Frontend View Container"])
async def serve_homepage():
    return FileResponse("static/index.html")

# --- API ENDPOINTS ---
@app.post("/api/v2/knowledge/upload", tags=["Knowledge Ingestion Studio"])
async def upload_knowledge_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        extracted_text = extract_text_from_file(file_path, file.filename)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Document text extraction yielded empty profile.")
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = text_splitter.split_text(extracted_text)

        if not chunks:
            raise HTTPException(
            status_code=400,
            detail="No semantic chunks generated from uploaded document."
            )
        #embeddings = embedding_model.encode(chunks).astype("float32") # type: ignore
        embeddings = embedding_model.encode(chunks)
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                embeddings=[
                    embeddings[i].tolist()
                ],

                metadatas=[{
                    "source": file.filename,
                    "chunk_id": i
                }],

                ids=[str(uuid.uuid4())]
            )
        return JSONResponse({"status": "Success", "filename": file.filename, "chunks_processed": len(chunks)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Updated Text Chat Endpoint (Now blazing fast) ---
@app.post("/api/v2/agent/text-chat", tags=["Agent Core Interaction Stack"])
async def handle_text_interaction(payload: TextChatRequest):
    try:
        # 1. Get the LLM/RAG response instantly
        execution_packet = await process_intent_routing_workflow(
            payload.text_query
        )
        
        # 2. Return it to the frontend immediately without waiting for TTS
        return JSONResponse(execution_packet)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- New Dedicated TTS Generation Endpoint ---
class TTSGenerationRequest(BaseModel):
    text: str
@app.post("/api/v2/agent/voice-chat", tags=["Agent Core Interaction Stack"])
async def handle_voice_interaction(file: UploadFile = File(...)):
    try:
        temp_audio_path = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}_capture.wav")
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await file.read())
            
        transcript = await anyio.to_thread.run_sync(transcribe_audio, temp_audio_path)
        pipeline_output_packet = await process_intent_routing_workflow(transcript)       
        pipeline_output_packet["transcript"] = transcript
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        return JSONResponse(pipeline_output_packet)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.post("/api/v2/agent/generate-audio", tags=["Agent Core Interaction Stack"])
async def handle_tts_generation(payload: TTSGenerationRequest):
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        # Run the heavy audio processing in a background worker thread
        audio_file = await anyio.to_thread.run_sync(generate_f5_tts, payload.text)
        
        audio_url = f"/tts/{audio_file}" if audio_file else None
        return JSONResponse({"audio_url": audio_url})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("medvoice:app", host="127.0.0.1", port=8000, reload=True)
    

