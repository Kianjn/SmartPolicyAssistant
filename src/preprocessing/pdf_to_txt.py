import os
import pdfplumber
import re
from datetime import datetime

# Set paths
pdf_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "input")
txt_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
log_path = os.path.join(txt_folder, "extraction_log.txt")
os.makedirs(txt_folder, exist_ok=True)

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n +', '\n', text)
    return text.strip()

def extract_country_name(filename):
    return re.split(r"[_\-\.]", filename)[0]

success_count = 0
fail_count = 0
log_lines = [f"Extraction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(txt_folder, txt_filename)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""

                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    table_text = ""
                    for table in tables:
                        for row in table:
                            row_text = "\t".join([cell if cell else "" for cell in row])
                            table_text += row_text + "\n"
                        table_text += "\n"

                    body_text = page.extract_text() or ""
                    combined = body_text + "\n" + table_text
                    cleaned = clean_text(combined)
                    all_text += cleaned + "\n\n"

            all_text = clean_text(all_text)

            country_name = extract_country_name(filename)
            metadata = f"""=== METADATA ===
Filename: {filename}
Country: {country_name}
Date of Extraction: {datetime.now().strftime('%Y-%m-%d')}
Tool: pdfplumber
Notes: Auto-cleaned text + table content, figures omitted.
==================

"""

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(metadata + all_text)

            log_lines.append(f"[✓] SUCCESS: {filename} extracted to {txt_filename}")
            success_count += 1

        except Exception as e:
            log_lines.append(f"[!] ERROR: {filename} failed — {str(e)}")
            fail_count += 1

# Final summary
log_lines.append(f"\n--- SUMMARY ---")
log_lines.append(f"Total PDFs processed: {success_count + fail_count}")
log_lines.append(f"Successful: {success_count}")
log_lines.append(f"Failed: {fail_count}")

# Write log file
with open(log_path, "w", encoding="utf-8") as log_file:
    log_file.write("\n".join(log_lines))

print(f"\n[✓] Extraction complete. Log saved to: {log_path}")