from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from src.models import ClaimResult
from src.pdf_utils import is_valid_pdf, extract_pdf_pages
from src.pipeline import run_pipeline

app = FastAPI()


@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(...),
    file: UploadFile = File(...)
) -> ClaimResult:
    # Validate claim_id (task 7.2)
    if not claim_id or not claim_id.strip():
        raise HTTPException(status_code=422, detail="claim_id must be non-empty and non-whitespace")

    # Read and validate PDF (task 7.3)
    pdf_bytes = await file.read()
    if not is_valid_pdf(pdf_bytes):
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid PDF")

    # Extract pages and run pipeline (task 7.4)
    try:
        pdf_pages = extract_pdf_pages(pdf_bytes)
        result = run_pipeline(claim_id=claim_id.strip(), pdf_pages=pdf_pages)
        return result
    except Exception as e:
        # task 7.5
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
