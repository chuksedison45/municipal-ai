import os
import re
import pdfplumber
from pathlib import Path

# --- CONFIGURATION ---
PDF_PATH = Path("source_data/test_file.pdf")
OUTPUT_FILE = Path("full_text_ocr.txt")

class MunicipalIngestor:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def extract_clean_text(self):
        """
        Extracts text using safe page boundaries and high tolerance to
        capture characters near margins without triggering ValueErrors.
        """
        if not self.file_path.exists():
            print(f"❌ Error: {self.file_path} not found.")
            return ""

        extracted_pages = []

        print(f"📖 Processing {self.file_path}...")
        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                width = page.width
                height = page.height
                midpoint = width / 2

                # SAFE BOUNDARIES: Clamping to 0 and page width to avoid ValueError
                # We use a slight overlap (2 points) at the midpoint to ensure no text is lost in the gutter.
                left_bbox = (0, 0, midpoint + 2, height)
                right_bbox = (midpoint - 2, 0, width, height)

                # TOLERANCE: x_tolerance=3 catches characters that might be
                # physically near the edge but technically offset in the PDF's text layer.
                left_text = page.crop(left_bbox).extract_text(x_tolerance=3, y_tolerance=3) or ""
                right_text = page.crop(right_bbox).extract_text(x_tolerance=3, y_tolerance=3) or ""

                extracted_pages.append(left_text + "\n" + right_text)

        full_text = "\n".join(extracted_pages)
        return self._post_process(full_text)

    def _post_process(self, text: str) -> str:
        """Removes page numbers and artifacts while preserving legal structure."""

        # 1. REMOVE ARTIFACTS: Cleans backslashes and source markers
        text = re.sub(r"\\", '', text)


        # 2. REMOVE PAGE NUMBERS: Matches standalone numbers like 409 or 410
        # Matches lines that contain only a 1-4 digit number.
        text = re.sub(r'(?m)^\s*\d{1,4}\s*$', '', text)

        # 3. FIX HYPHENATION: Joins words split across lines [cite: 6, 10, 12, 14, 15, 22]
        # e.g., 'exer- cise' -> 'exercise', 'emer- gency' -> 'emergency'
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # 4. NORMALIZE WHITESPACE: Ensures consistent spacing for the vector DB
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

def main():
    ingestor = MunicipalIngestor(PDF_PATH)
    clean_text = ingestor.extract_clean_text()

    if clean_text:
        # Exporting specifically to 'full_text_ocr.txt' for Lab 2
        OUTPUT_FILE.write_text(clean_text, encoding="utf-8")
        print(f"✅ Success! Clean text exported to '{OUTPUT_FILE}'")

        # Verification of key sections from the provided PDF [cite: 5, 24]
        print("\n--- Content Verification ---")
        if "12.12.010" in clean_text:
            print("Found Section 12.12.010 [cite: 5]")
        if "12.12.020" in clean_text:
            print("Found Section 12.12.020 [cite: 24, 28]")

        # Verify specific legal citations were preserved [cite: 23, 33]
        if "Prior code §" in clean_text:
            print("Found 'Prior code' citations [cite: 23, 33]")

        print(f"\nPreview (First 300 chars):\n{'-'*20}\n{clean_text[:300]}\n{'-'*20}")
    else:
        print("❌ Extraction failed.")

if __name__ == "__main__":
    main()
