import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from typing import Dict, Any

def extract_text_pdfplumber(pdf_path: str) -> str:
    """Extract plain text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)

def extract_layout_features(pdf_path: str) -> Dict[str, Any]:
    """Extract layout features using PyMuPDF for multimodal angle."""
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    font_sizes = []
    section_count = 0
    total_chars = 0
    bold_chars = 0

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
                        total_chars += len(span["text"])
                        if "bold" in span["font"].lower():
                            bold_chars += len(span["text"])
                        # Heuristic: large font = section header
                        if span["size"] > 14:
                            section_count += 1
    doc.close()

    return {
        "num_pages": num_pages,
        "avg_font_size": sum(font_sizes) / len(font_sizes) if font_sizes else 0,
        "max_font_size": max(font_sizes) if font_sizes else 0,
        "min_font_size": min(font_sizes) if font_sizes else 0,
        "font_size_variance": pd.Series(font_sizes).var() if font_sizes else 0,
        "section_count": section_count,
        "bold_ratio": bold_chars / total_chars if total_chars > 0 else 0,
    }

def parse_all_resumes(pdf_dir: str, output_path: str) -> pd.DataFrame:
    """Parse all resume PDFs into a structured DataFrame."""
    records = []
    pdf_dir = Path(pdf_dir)

    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            text = extract_text_pdfplumber(str(pdf_path))
            features = extract_layout_features(str(pdf_path))
            records.append({
                "id": pdf_path.stem,
                "text": text,
                "layout_features": features,
                **features  # flatten for easy access
            })
        except Exception as e:
            print(f"Failed to parse {pdf_path.name}: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    return df