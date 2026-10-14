"""
AI Job Hunt OS — Enhanced Edition v4.2
Streamlit app tracking job applications with human-in-the-loop editing,
Groq prompt cache pricing calculations, dynamic standard file naming,
local downloads, and Supabase Storage bucket sync.

Run:
    streamlit run app.py
"""

import io
import os
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker
from groq import Groq
from openai import OpenAI

# Safe import for rich text editor component
try:
    from streamlit_lexical_extended import streamlit_lexical
    HAS_LEXICAL = True
except ImportError:
    HAS_LEXICAL = False

# Document generation packages
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Supabase Storage Client
try:
    from supabase import create_client, Client
except ImportError:
    Client = None

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Job Hunt OS",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

def _get_secret(key: str, default=None):
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return val
    except ImportError:
        pass

    val = os.environ.get(key)
    if val:
        return val

    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass

    return default

GROQ_API_KEY       = _get_secret("GROQ_API_KEY")
SEALION_API_KEY    = _get_secret("SEALION_API_KEY")
HF_TOKEN           = _get_secret("HF_TOKEN")
SUPABASE_DB_URL    = _get_secret("SUPABASE_DB_URL")
SUPABASE_URL       = _get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY  = _get_secret("SUPABASE_ANON_KEY")

if not GROQ_API_KEY and not SEALION_API_KEY:
    st.error(
        "❌ No API keys found. Set at least one of GROQ_API_KEY or SEALION_API_KEY "
        "in your `.env` file or Streamlit secrets."
    )
    st.stop()

if not SUPABASE_DB_URL:
    st.error(
        "❌ No SUPABASE_DB_URL found. Set your Supabase PostgreSQL connection string "
        "in your `.env` file or Streamlit secrets."
    )
    st.stop()

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SEALION_BASE_URL = "https://api.sea-lion.ai/v1"
sealion_client = (
    OpenAI(api_key=SEALION_API_KEY, base_url=SEALION_BASE_URL)
    if SEALION_API_KEY else None
)

STAGES = [
    {"id": "Applied",      "name": "Applied",      "color": "#3498db"},
    {"id": "Acknowledged", "name": "Acknowledged", "color": "#f39c12"},
    {"id": "Screening",    "name": "Screening",    "color": "#9b59b6"},
    {"id": "Interviewing", "name": "Interviewing", "color": "#e67e22"},
    {"id": "Offer",        "name": "Offer",        "color": "#27ae60"},
    {"id": "Rejected",     "name": "Rejected",     "color": "#e74c3c"},
]

# Database Connection Driver Setup
if SUPABASE_DB_URL.startswith("postgres://"):
    DATABASE_URL = SUPABASE_DB_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif SUPABASE_DB_URL.startswith("postgresql://"):
    DATABASE_URL = SUPABASE_DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
else:
    DATABASE_URL = SUPABASE_DB_URL

# Model Registry
MODEL_REGISTRY = {
    "Sea-Lion v4 27B": {
        "provider": "sealion",
        "model_id": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
        "description": "Gemma 3 27B fine-tuned for SEA languages by AI Singapore.",
    },
    "Sea-Lion v4 32B (Qwen)": {
        "provider": "sealion",
        "model_id": "aisingapore/Qwen-SEA-LION-v4-32B-IT",
        "description": "Qwen3-32B fine-tuned for SEA tasks.",
    },
    "Groq: GPT-OSS 120B": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "description": "OpenAI's open-weight MoE, 120B params, reasoning-capable.",
    },
    "Groq: GPT-OSS 20B": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "description": "Smaller, faster OpenAI open-weight model.",
    },
}

DEFAULT_MODEL_KEY = "Sea-Lion v4 27B"

# Model Pricing (per 1M tokens, USD)
MODEL_PRICING = {
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "cached_input": 0.075,  # 50% discount for cached input
        "output": 0.60,
    },
    "openai/gpt-oss-20b": {
        "input": 0.075,
        "cached_input": 0.037,
        "output": 0.30,
    },
    "aisingapore/Gemma-SEA-LION-v4-27B-IT": {
        "input": 0.0, "cached_input": 0.0, "output": 0.0
    },
    "aisingapore/Qwen-SEA-LION-v4-32B-IT": {
        "input": 0.0, "cached_input": 0.0, "output": 0.0
    },
}

def get_pricing(model_id: str) -> dict:
    return MODEL_PRICING.get(
        model_id,
        {"input": 0.0, "cached_input": 0.0, "output": 0.0}
    )

def compute_cost_usd(
    model_id: str, 
    input_tokens: int, 
    output_tokens: int, 
    cached_tokens: int = 0
) -> float:
    """Calculates total request cost taking cached prompt tokens into account."""
    p = get_pricing(model_id)
    
    uncached_input_tokens = max(0, input_tokens - cached_tokens)
    
    uncached_cost = (uncached_input_tokens / 1_000_000) * p.get("input", 0.0)
    cached_cost = (cached_tokens / 1_000_000) * p.get("cached_input", p.get("input", 0.0) * 0.5)
    output_cost = (output_tokens / 1_000_000) * p.get("output", 0.0)
    
    return uncached_cost + cached_cost + output_cost

# ---------------------------------------------------------------------------
# 2. DATABASE LAYER
# ---------------------------------------------------------------------------

Base = declarative_base()

class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True)
    company = Column(String, nullable=False)
    role = Column(String, nullable=False)
    job_url = Column(String, default="")
    status = Column(String, default="Applied")
    
    # Custom Naming Metadata
    prefix = Column(String, default="PWA")
    seq_num = Column(Integer, default=286)
    
    # Storage and Document File References
    resume_filename = Column(String, default="")
    resume_storage_url = Column(String, default="")
    cover_filename = Column(String, default="")
    cover_storage_url = Column(String, default="")

    job_description = Column(Text, default="")
    resume_text = Column(Text, default="")
    cover_letter = Column(Text, default="")
    match_score = Column(String, default="")
    applied_date = Column(DateTime, default=datetime.now)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id = Column(Integer, primary_key=True)
    called_at = Column(DateTime, default=datetime.now, index=True)
    provider = Column(String)
    model_key = Column(String)
    model_id = Column(String)
    operation = Column(String)
    input_tokens = Column(Integer, default=0)
    cached_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    latency_seconds = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    success = Column(Integer, default=1)
    error_message = Column(Text, default="")

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


@st.cache_resource
def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=120,
        pool_size=5,
        max_overflow=5
    )

@st.cache_resource
def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)

def db_session():
    return get_session_factory()()

# ---------------------------------------------------------------------------
# 3. HELPER UTILITIES
# ---------------------------------------------------------------------------

def render_editor(default_text: str, key: str) -> str:
    """Renders WYSIWYG editor if available, otherwise falls back to st.text_area."""
    if HAS_LEXICAL:
        edited = streamlit_lexical(value=default_text, key=key)
        return edited if edited is not None else default_text
    else:
        return st.text_area("Edit Document Content", value=default_text, height=300, key=key)

def get_next_sequence_number() -> int:
    with db_session() as db:
        max_seq = db.query(func.max(Application.seq_num)).scalar()
        return (max_seq or 285) + 1

def format_doc_filename(
    doc_type: str, role_title: str, company_name: str, seq_number: int, prefix: str = "PWA"
) -> str:
    clean_role = re.sub(r'[\\/*?:"<>|]', "", role_title).strip()
    clean_company = re.sub(r'[\\/*?:"<>|]', "", company_name).strip()
    formatted_seq = f"{int(seq_number):05d}"
    date_str = datetime.now().strftime("%d%m%y")
    
    if doc_type == "resume":
        doc_label = "Resume"
        suffix = "CV"
    else:
        doc_label = "Cover Letter"
        suffix = "CL"
        
    return f"{prefix}{formatted_seq}_{doc_label} for {clean_role}_{clean_company}_{date_str}_{suffix}"

def generate_docx_bytes(content_text: str) -> bytes:
    doc = Document()
    for paragraph in content_text.split('\n'):
        if paragraph.strip():
            doc.add_paragraph(paragraph)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

def generate_pdf_bytes(content_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14
    
    story = []
    for line in content_text.split('\n'):
        if line.strip():
            clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(clean_line, normal_style))
            story.append(Spacer(1, 6))
            
    doc.build(story)
    return buffer.getvalue()

def upload_bytes_to_supabase_bucket(
    file_bytes: bytes, bucket_name: str, destination_path: str, file_type: str = "application/pdf"
) -> str:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.warning("Supabase URL or Anon Key missing. File saved locally only.")
        return ""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        supabase.storage.from_(bucket_name).upload(
            path=destination_path,
            file=file_bytes,
            file_options={"content-type": file_type, "upsert": "true"}
        )
        return supabase.storage.from_(bucket_name).get_public_url(destination_path)
    except Exception as e:
        st.error(f"Failed to upload file to Supabase: {e}")
        return ""

# ---------------------------------------------------------------------------
# 4. AI & SERVICE LAYER
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

def get_active_model() -> tuple[str, str, str]:
    key = st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY)
    entry = MODEL_REGISTRY.get(key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    return entry["provider"], entry["model_id"], key

def load_llm_for(provider: str, model_id: str):
    if provider == "groq" and GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_id, temperature=0.7, api_key=GROQ_API_KEY), f"Groq ({model_id})"
    if provider == "sealion" and SEALION_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=0.7, api_key=SEALION_API_KEY, base_url=SEALION_BASE_URL), f"Sea Lion ({model_id})"
    return None, "LLM Key Missing"

def _extract_usage_from_langchain(result) -> tuple[int, int, int]:
    um = getattr(result, "usage_metadata", None) or {}
    rm = getattr(result, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}

    input_tokens = int(um.get("input_tokens") or tu.get("prompt_tokens") or 0)
    output_tokens = int(um.get("output_tokens") or tu.get("completion_tokens") or 0)
    
    ptd = tu.get("prompt_tokens_details") or {}
    cached_tokens = int(ptd.get("cached_tokens") or um.get("input_token_details", {}).get("cache_read") or 0)

    return input_tokens, output_tokens, cached_tokens

def svc_log_llm_usage(*, provider, model_key, model_id, operation, input_tokens, output_tokens, cached_tokens=0, latency_seconds=0.0, success=True, error_message=""):
    cost = compute_cost_usd(model_id, input_tokens, output_tokens, cached_tokens) if success else 0.0
    with db_session() as db:
        row = LLMUsage(
            provider=provider, model_key=model_key, model_id=model_id, operation=operation,
            input_tokens=input_tokens, cached_tokens=cached_tokens, output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens, latency_seconds=float(latency_seconds or 0.0),
            cost_usd=cost, success=1 if success else 0, error_message=error_message or "",
        )
        db.add(row)
        db.commit()

def call_llm(prompt: str, operation: str = "generic") -> str:
    provider, model_id, label = get_active_model()
    llm, status = load_llm_for(provider, model_id)

    if llm is None:
        svc_log_llm_usage(provider=provider, model_key=label, model_id=model_id, operation=operation, input_tokens=0, output_tokens=0, cached_tokens=0, latency_seconds=0.0, success=False, error_message=str(status))
        return f"⚠️ {status}"

    start = time.time()
    try:
        result = llm.invoke(prompt)
        latency = time.time() - start
        in_tok, out_tok, cached_tok = _extract_usage_from_langchain(result)
        svc_log_llm_usage(provider=provider, model_key=label, model_id=model_id, operation=operation, input_tokens=in_tok, output_tokens=out_tok, cached_tokens=cached_tok, latency_seconds=latency, success=True)
        return getattr(result, "content", str(result))
    except Exception as e:
        latency = time.time() - start
        svc_log_llm_usage(provider=provider, model_key=label, model_id=model_id, operation=operation, input_tokens=0, output_tokens=0, cached_tokens=0, latency_seconds=latency, success=False, error_message=str(e))
        return f"⚠️ LLM call failed: {e}"

def tailor_resume(resume_text: str, job_description: str) -> str:
    prompt = f"Rewrite the resume to emphasize experience relevant to this job description:\n\nJOB:\n{job_description}\n\nRESUME:\n{resume_text}"
    return call_llm(prompt, operation="tailor_resume")

def generate_cover_letter(resume_text: str, job_description: str, company: str, role: str) -> str:
    prompt = f"Write a cover letter for {role} at {company}:\n\nJOB:\n{job_description}\n\nRESUME:\n{resume_text}"
    return call_llm(prompt, operation="cover_letter")

def extract_pdf_text(uploaded_file) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        return "".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

def svc_list_applications():
    with db_session() as db:
        return [a.to_dict() for a in db.query(Application).order_by(Application.last_updated.desc()).all()]

def svc_create_application(**kwargs):
    with db_session() as db:
        app = Application(**kwargs)
        db.add(app)
        db.commit()
        db.refresh(app)
        return app.id

def svc_update_application_files(app_id: int, **kwargs):
    with db_session() as db:
        app = db.query(Application).filter(Application.id == app_id).first()
        if app:
            for k, v in kwargs.items():
                setattr(app, k, v)
            db.commit()

# ---------------------------------------------------------------------------
# 5. UI & EDITING WORKSPACE
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("💼 AI Job Blessed Hub")
        st.caption("AI-powered job search tracker")
        st.markdown("---")
        st.markdown("**👤 Lester**")
        st.caption("lesterchia1@hotmail.com")
        st.markdown("---")

        nav = st.radio("Navigation", ["📊 Tracker", "➕ New Application", "📈 Analytics"], index=1)
        st.markdown("---")
        
        options = list(MODEL_REGISTRY.keys())
        selected = st.selectbox("Select Model", options, index=0)
        st.session_state["selected_model_key"] = selected
        
    return nav

def view_new_application():
    st.title("➕ New Application & AI Document Hub")
    st.caption("Customize filenames, select selective AI tasks, edit, and sync to Supabase.")

    auto_seq = get_next_sequence_number()

    with st.form("new_app_form"):
        st.subheader("1. Job Application Details")
        c1, c2 = st.columns(2)
        company = c1.text_input("Company Name *", "SUNTORY")
        role = c2.text_input("Role Title *", "AI Product Analyst - 1 Year Contract")
        job_url = st.text_input("Job URL (optional)")
        job_description = st.text_area("Job Description", height=150)
        resume_file = st.file_uploader("Upload Base Resume (PDF)", type=["pdf"])

        st.markdown("---")
        st.subheader("2. Standard File Naming Rules")
        fc1, fc2 = st.columns(2)
        prefix = fc1.text_input("Prefix", value="PWA")
        seq_num = fc2.number_input("Sequence Number", min_value=1, value=auto_seq, step=1)

        st.markdown("---")
        st.subheader("3. Generation Scope")
        gen_choice = st.radio(
            "Select what you want the AI to generate:",
            options=[
                "a. Resume only",
                "b. Cover letter only",
                "c. Resume and Cover letter",
                "d. Nil (Save application details only)"
            ],
            index=2
        )

        submitted = st.form_submit_button("🚀 Process Application Choice", type="primary")

    if submitted:
        if not company or not role:
            st.error("Company Name and Role Title are required.")
            return

        base_resume_text = extract_pdf_text(resume_file) if resume_file else ""

        should_gen_resume = gen_choice in ["a. Resume only", "c. Resume and Cover letter"]
        should_gen_cover = gen_choice in ["b. Cover letter only", "c. Resume and Cover letter"]

        tailored_resume = ""
        cover_letter = ""

        if should_gen_resume:
            if base_resume_text and job_description:
                with st.spinner("Tailoring Resume..."):
                    tailored_resume = tailor_resume(base_resume_text, job_description)
            else:
                st.warning("Skipped Resume generation: PDF resume or job description missing.")

        if should_gen_cover:
            ref_text = tailored_resume if tailored_resume else base_resume_text
            if ref_text and job_description:
                with st.spinner("Drafting Cover Letter..."):
                    cover_letter = generate_cover_letter(ref_text, job_description, company, role)
            else:
                st.warning("Skipped Cover Letter generation: Base resume or job description missing.")

        app_id = svc_create_application(
            company=company, role=role, job_url=job_url,
            job_description=job_description, resume_text=tailored_resume,
            cover_letter=cover_letter, prefix=prefix, seq_num=seq_num, status="Applied"
        )

        st.session_state["active_app"] = {
            "app_id": app_id, "company": company, "role": role,
            "prefix": prefix, "seq_num": seq_num,
            "tailored_resume": tailored_resume, "cover_letter": cover_letter,
        }
        st.success(f"Application #{app_id} created successfully!")

    app_data = st.session_state.get("active_app")
    if app_data:
        st.markdown("---")
        st.header("✏️ Human-in-the-Loop Edit & Export Workspace")

        if app_data.get("tailored_resume"):
            st.subheader("📄 Edit Tailored Resume")
            res_filename = format_doc_filename("resume", app_data["role"], app_data["company"], app_data["seq_num"], app_data["prefix"])
            st.caption(f"Standardized Filename: `{res_filename}`")

            edited_resume = render_editor(app_data["tailored_resume"], key="resume_editor")

            res_docx = generate_docx_bytes(edited_resume)
            res_pdf = generate_pdf_bytes(edited_resume)

            rc1, rc2, rc3 = st.columns(3)
            rc1.download_button("📥 Download Resume (.docx)", res_docx, f"{res_filename}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            rc2.download_button("📥 Download Resume (.pdf)", res_pdf, f"{res_filename}.pdf", "application/pdf")
            if rc3.button("☁️ Save Resume to Supabase Bucket"):
                dest = f"resumes/{res_filename}.pdf"
                public_url = upload_bytes_to_supabase_bucket(res_pdf, "documents", dest)
                svc_update_application_files(app_data["app_id"], resume_filename=f"{res_filename}.pdf", resume_storage_url=public_url)
                st.success(f"Uploaded! Filename stored in DB: `{res_filename}.pdf`")

        if app_data.get("cover_letter"):
            st.markdown("---")
            st.subheader("✉️ Edit Cover Letter")
            cov_filename = format_doc_filename("cover_letter", app_data["role"], app_data["company"], app_data["seq_num"], app_data["prefix"])
            st.caption(f"Standardized Filename: `{cov_filename}`")

            edited_cover = render_editor(app_data["cover_letter"], key="cover_editor")

            cov_docx = generate_docx_bytes(edited_cover)
            cov_pdf = generate_pdf_bytes(edited_cover)

            cc1, cc2, cc3 = st.columns(3)
            cc1.download_button("📥 Download Cover Letter (.docx)", cov_docx, f"{cov_filename}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            cc2.download_button("📥 Download Cover Letter (.pdf)", cov_pdf, f"{cov_filename}.pdf", "application/pdf")
            if cc3.button("☁️ Save Cover Letter to Supabase Bucket"):
                dest = f"cover_letters/{cov_filename}.pdf"
                public_url = upload_bytes_to_supabase_bucket(cov_pdf, "documents", dest)
                svc_update_application_files(app_data["app_id"], cover_filename=f"{cov_filename}.pdf", cover_storage_url=public_url)
                st.success(f"Uploaded! Filename stored in DB: `{cov_filename}.pdf`")

def main():
    nav = render_sidebar()
    if nav == "➕ New Application":
        view_new_application()
    else:
        st.title("📊 Applications Tracker")
        apps = svc_list_applications()
        st.dataframe(pd.DataFrame(apps) if apps else pd.DataFrame())

if __name__ == "__main__":
    main()
