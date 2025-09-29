import os
import io
import json
import re
import base64
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
from scipy import stats

load_dotenv()

# ======================
# CONFIG
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))

app = FastAPI()


# ===== Gemini API Calls =====
def call_gemini(prompt: str):
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        out = resp.json()
        return out["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini API call failed: {e}"


def call_gemini_image(prompt: str, img_base64: str, mime_type: str):
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY,
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": img_base64}},
                    {"text": prompt},
                ]
            }
        ]
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=90)
        resp.raise_for_status()
        out = resp.json()
        return out["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini Vision API call failed: {e}"


# ===== URL Extraction =====
def extract_urls(text: str):
    url_pattern = r'https?://[^\s"\'\)]+|www\.[^\s"\'\)]+'
    urls = re.findall(url_pattern, text)
    cleaned_urls = []
    for url in urls:
        url = url.strip().strip('"').strip("'").strip(')').strip(',').strip('.')
        if not url.startswith('http'):
            url = 'https://' + url
        cleaned_urls.append(url)
    return cleaned_urls


# ===== Web Scraping =====
def scrape_website(url: str):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        tables = []
        try:
            tables = pd.read_html(io.StringIO(response.text), flavor='lxml')
        except:
            try:
                tables = pd.read_html(response.text, flavor='html5lib')
            except:
                try:
                    tables = pd.read_html(response.text)
                except:
                    return None

        if not tables:
            return None
        return max(tables, key=len)

    except Exception:
        return None


# ===== Analyze Questions =====
def analyze_questions(q_text: str):
    try:
        urls = extract_urls(q_text)
        json_requested = "json" in q_text.lower() or "array" in q_text.lower()
        all_responses = []

        if not urls:
            gemini_answer = call_gemini(q_text)
            return [gemini_answer]

        for url in urls:
            try:
                df = scrape_website(url)
                if df is None or df.empty:
                    gemini_answer = call_gemini(f"Visit this URL: {url} and answer these questions: {q_text}")
                    all_responses.append(gemini_answer)
                    continue

                df = df.dropna(how='all')
                df.columns = [str(col).strip() for col in df.columns]

                df_sample = df.head(5).to_string()
                data_summary = f"""
                Data from {url}:
                - Shape: {df.shape[0]} rows, {df.shape[1]} columns
                - Columns: {list(df.columns)}
                - Sample data:
                {df_sample}
                Questions:
                {q_text}
                """
                gemini_answer = call_gemini(data_summary)
                all_responses.append(gemini_answer)

            except Exception as e:
                all_responses.append(f"Error processing URL {url}: {e}")

        return all_responses

    except Exception as e:
        return [f"Error analyzing questions: {str(e)}"]


# ===== API Endpoint =====
@app.post("/api/analyse")
async def api_analyse(
    files: list[UploadFile] = File(None),
    manual_question: str = Form(None)
):
    results = []
    try:
        if not files and not manual_question:
            return JSONResponse(content={"error": "Please upload at least one file or provide a question."}, status_code=400)

        for uploaded_file in files or []:
            file_extension = uploaded_file.filename.split(".")[-1].lower()
            file_content = await uploaded_file.read()

            if file_extension in ["png", "jpg", "jpeg"]:
                img_base64 = base64.b64encode(file_content).decode("utf-8")
                prompt = "Extract all textual content from this image as one plain text block."
                ocr_text = call_gemini_image(prompt, img_base64, mime_type=uploaded_file.content_type)
                q_text = ocr_text
            else:
                q_text = file_content.decode("utf-8")

            if manual_question:
                q_text = f"{manual_question}\n\n{q_text}"

            q_text = q_text.replace("\r", "").strip()
            q_text = re.sub(r"\s+", " ", q_text)

            answers = analyze_questions(q_text)
            results.append({"filename": uploaded_file.filename, "answers": answers})

        if not files and manual_question:
            q_text = manual_question
            q_text = q_text.replace("\r", "").strip()
            q_text = re.sub(r"\s+", " ", q_text)
            answers = analyze_questions(q_text)
            results.append({"filename": "manual_input", "answers": answers})

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": f"Error analyzing your request: {str(e)}"}, status_code=500)


# ===== Run FastAPI =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=FASTAPI_PORT)
