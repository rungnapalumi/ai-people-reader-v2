#!/usr/bin/env python3
"""
Convert a DOCX file to PDF using LibreOffice (or docx2pdf on Windows if Word is installed).
Use this for manual testing when local web PDF works but deployed PDF has Thai font issues.

Usage:
  python scripts/docx_to_pdf.py [docx_path] [-o output.pdf] [--use-libreoffice | --use-docx2pdf]
  python scripts/docx_to_pdf.py test_report_th.docx -o output.pdf

Edit the variables below and run (no terminal args needed) for quick testing.
"""
# --- Edit these for quick run ---
INPUT_DOCX = "test_report_th.docx"  # or "assets/Presentation_Analysis_Report_2026-03-06_TH.docx"
OUTPUT_PDF = "output.pdf"  # Leave empty to use same path with .pdf extension
USE_DOCX2PDF = False  # True = use Word (Windows), False = use LibreOffice
# -----------------
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def _find_libreoffice() -> str:
    """Find LibreOffice/soffice binary (cross-platform)."""
    for name in ("libreoffice", "soffice"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    # Linux/macOS paths
    for path in (
        "/usr/bin/libreoffice",
        "/usr/bin/soffice",
        "/usr/lib/libreoffice/program/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ):
        if os.path.exists(path):
            return path
    # Windows paths
    for path in (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ):
        if os.path.exists(path):
            return path
    return ""


def convert_with_libreoffice(docx_path: str, pdf_path: str, timeout: int = 180) -> None:
    """Convert DOCX to PDF using LibreOffice headless."""
    lo_bin = _find_libreoffice()
    if not lo_bin:
        raise RuntimeError(
            "LibreOffice not found. Install it from https://www.libreoffice.org/download/"
        )
    out_dir = os.path.dirname(pdf_path)
    cmd = [
        lo_bin,
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to",
        "pdf:writer_pdf_Export",
        "--outdir",
        out_dir,
        os.path.abspath(docx_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, timeout=timeout
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"LibreOffice failed (rc={proc.returncode}): {err[:500]}")
    # LibreOffice uses input filename with .pdf; find the output
    stem = os.path.splitext(os.path.basename(docx_path))[0]
    expected = os.path.join(out_dir, stem + ".pdf")
    actual = expected if os.path.exists(expected) else None
    if not actual:
        for f in os.listdir(out_dir):
            if f.lower().endswith(".pdf"):
                actual = os.path.join(out_dir, f)
                break
    if not actual or not os.path.exists(actual):
        raise RuntimeError("LibreOffice succeeded but PDF output not found")
    if actual != pdf_path:
        os.rename(actual, pdf_path)


def convert_with_docx2pdf(docx_path: str, pdf_path: str) -> None:
    """Convert DOCX to PDF using docx2pdf (Windows only, requires Microsoft Word)."""
    try:
        from docx2pdf import convert as docx2pdf_convert
    except ImportError:
        raise RuntimeError("docx2pdf not installed. Run: pip install docx2pdf")
    docx2pdf_convert(os.path.abspath(docx_path), os.path.abspath(pdf_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert DOCX to PDF")
    parser.add_argument("docx", nargs="?", default=INPUT_DOCX, help="Input DOCX file path")
    parser.add_argument("-o", "--output", default=OUTPUT_PDF or None, help="Output PDF path")
    parser.add_argument("--timeout", type=int, default=180, help="LibreOffice timeout in seconds")
    parser.add_argument("--use-docx2pdf", action="store_true", default=USE_DOCX2PDF, help="Use docx2pdf (Windows + Word)")
    parser.add_argument("--use-libreoffice", action="store_true", help="Force LibreOffice instead of docx2pdf")
    args = parser.parse_args()
    docx_path = args.docx
    if not docx_path:
        print("Error: No input file. Set INPUT_DOCX at top of script or pass as argument.", file=sys.stderr)
        return 1
    if not os.path.exists(docx_path):
        print(f"Error: File not found: {docx_path}", file=sys.stderr)
        return 1
    pdf_path = args.output if args.output else os.path.splitext(docx_path)[0] + ".pdf"
    use_docx2pdf = args.use_docx2pdf and not args.use_libreoffice
    try:
        if use_docx2pdf:
            print("Using docx2pdf (Microsoft Word)...")
            convert_with_docx2pdf(docx_path, pdf_path)
        else:
            print("Using LibreOffice...")
            convert_with_libreoffice(docx_path, pdf_path, timeout=args.timeout)
        print(f"Done: {pdf_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not use_docx2pdf and _find_libreoffice():
            print("LibreOffice was found but conversion failed.", file=sys.stderr)
        elif not use_docx2pdf:
            print("Tip: Install LibreOffice, or try: pip install docx2pdf && python ... --use-docx2pdf (requires Word on Windows)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
