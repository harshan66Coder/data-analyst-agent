import os
import io
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import streamlit as st
import threading
import time
import base64
import re
from PIL import Image
from scipy import stats
from fastapi import Form
from dotenv import load_dotenv
load_dotenv()

# ======================
# CONFIG
# ======================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# ======================
# FastAPI APP
# ======================
app = FastAPI()




# ===== Gemini API Calls =====
def call_gemini(prompt: str):
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    print("===== Gemini Payload =====")
    print(data)   # 🔍 Print the payload being sent
    print("===========================")
    
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        out = resp.json()
        answer = out["candidates"][0]["content"]["parts"][0]["text"]
        print("Answer:", answer)
        if not answer or "Unable to retrieve data" in answer:
            return "Gemini API did not return a usable answer."
        return answer
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
        answer = out["candidates"][0]["content"]["parts"][0]["text"]
        return answer
    except Exception as e:
        return f"Gemini Vision API call failed: {e}"

# ===== Enhanced URL Extraction =====
def extract_urls(text: str):
    """Extract all URLs from text"""
    url_pattern = r'https?://[^\s"\'\)]+|www\.[^\s"\'\)]+'
    urls = re.findall(url_pattern, text)
    # Clean URLs by removing trailing punctuation and quotes
    cleaned_urls = []
    for url in urls:
        url = url.strip().strip('"').strip("'").strip(')').strip(',').strip('.')
        if not url.startswith('http'):
            url = 'https://' + url
        cleaned_urls.append(url)
    return cleaned_urls

# ===== Enhanced Web Scraping =====
def scrape_website(url: str):
    """Enhanced web scraping with better error handling"""
    try:
        print(f"Scraping URL: {url}")
        url = url.strip().strip('"').strip("'")  # Remove extra spaces and quotes

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        # Try different methods to parse tables
        tables = []
        try:
            tables = pd.read_html(io.StringIO(response.text), flavor='lxml')
        except:
            try:
                tables = pd.read_html(response.text, flavor='html5lib')
            except:
                try:
                    tables = pd.read_html(response.text)
                except Exception as e:
                    print(f"Could not parse tables: {e}")
                    return None

        if not tables:
            print("No tables found on the page")
            return None
            
        # Return the largest table (most likely to contain the main data)
        largest_table = max(tables, key=len)
        print(f"Found {len(tables)} tables, using largest with {len(largest_table)} rows")
        return largest_table
        
    except Exception as e:
        print(f"Scraping error: {e}")
        return None

# ===== Generate Scatterplot =====
def generate_scatterplot(df, x_col='Rank', y_col='Peak', max_size_bytes=100000):
    """Generate scatterplot with any two numeric columns"""
    try:
        def create_plot(figsize=(8, 5), dpi=100):
            plt.figure(figsize=figsize, dpi=dpi)
            df_clean = df.copy()
            
            # Check if columns exist
            if x_col not in df_clean.columns or y_col not in df_clean.columns:
                # Try to find suitable numeric columns
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    x_col_use, y_col_use = numeric_cols[0], numeric_cols[1]
                else:
                    plt.text(0.5, 0.5, 'No suitable numeric columns found', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Scatterplot Error')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
                    buf.seek(0)
                    plt.close()
                    img_bytes = buf.getvalue()
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    return img_b64, len(img_bytes)
            else:
                x_col_use, y_col_use = x_col, y_col
            
            df_clean[x_col_use] = pd.to_numeric(df_clean[x_col_use], errors='coerce')
            df_clean[y_col_use] = pd.to_numeric(df_clean[y_col_use], errors='coerce')
            df_clean = df_clean.dropna(subset=[x_col_use, y_col_use])
            
            if len(df_clean) == 0:
                plt.text(0.5, 0.5, 'No numeric data available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'Scatterplot of {x_col_use} vs {y_col_use}')
            else:
                df_clean = df_clean.sort_values(x_col_use)
                plt.scatter(df_clean[x_col_use], df_clean[y_col_use], alpha=0.7, s=40,
                           color='blue', edgecolors='black', linewidth=0.5)
                
                # Add regression line if we have enough data points
                if len(df_clean) > 1:
                    try:
                        slope, intercept, r_value, _, _ = stats.linregress(df_clean[x_col_use], df_clean[y_col_use])
                        regression_line = slope * df_clean[x_col_use] + intercept
                        plt.plot(df_clean[x_col_use], regression_line, 'r--', linewidth=1.5,
                               label=f'Regression (r={r_value:.6f})')
                        plt.legend()
                    except:
                        pass  # Skip regression line if calculation fails
                
                plt.xlabel(x_col_use)
                plt.ylabel(y_col_use)
                plt.title(f'Scatterplot of {x_col_use} vs {y_col_use}')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            img_bytes = buf.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_b64, len(img_bytes)

        b64_img, size_bytes = create_plot()
        if size_bytes > max_size_bytes:
            b64_img, size_bytes = create_plot(figsize=(6, 4), dpi=80)
            if size_bytes > max_size_bytes:
                b64_img, size_bytes = create_plot(figsize=(4, 3), dpi=70)
        
        return f"data:image/png;base64,{b64_img}"
    except Exception as e:
        print(f"Plot generation error: {e}")
        return None



# ===== Analyze Questions =====
def analyze_questions(q_text: str, attachments=None):
    """Dynamic question analysis with support for any URL"""
    try:
        # Extract all URLs from the text
        urls = extract_urls(q_text)
        print(f"Found URLs: {urls}")
        
        # Check if JSON response is requested
        json_requested = "json" in q_text.lower() or "array" in q_text.lower()
        
        if not urls:
            # No URLs found, use Gemini for general analysis
            gemini_prompt = f"Analyze this request and provide the response in the format requested: {q_text}"
            gemini_answer = call_gemini(gemini_prompt)
            
            if json_requested:
                # Try to extract or create JSON from Gemini's response
                try:
                    json_match = re.search(r'\[.*\]', gemini_answer, re.DOTALL)
                    if json_match:
                        return [json_match.group()]
                    else:
                        # Create a simple JSON array
                        return [json.dumps([gemini_answer])]
                except:
                    return [json.dumps([gemini_answer])]
            return [gemini_answer]
        
        # Process each URL and combine results
        all_responses = []
        
        for url in urls:
            try:
                df = scrape_website(url)
                
                if df is None or df.empty:
                    # If we couldn't scrape data, ask Gemini about the URL
                    url_prompt = f"Visit this URL: {url} and answer these questions: {q_text}"
                    gemini_answer = call_gemini(url_prompt)
                    
                    if json_requested:
                        # Try to extract JSON from the response
                        try:
                            json_match = re.search(r'\[.*\]', gemini_answer, re.DOTALL)
                            if json_match:
                                all_responses.append(json_match.group())
                            else:
                                all_responses.append(json.dumps([gemini_answer]))
                        except:
                            all_responses.append(json.dumps([gemini_answer]))
                    else:
                        all_responses.append(gemini_answer)
                    continue
                
                # We have data from the URL, create a summary for Gemini
                df = df.dropna(how='all')  # Clean up empty rows
                df.columns = [str(col).strip() for col in df.columns]  # Clean column names
                
                # Extract columns that might be useful for specific questions
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Analyze what kind of data we have
                rank_col = next((col for col in df.columns if 'rank' in str(col).lower()), None)
                gross_col = next((col for col in df.columns if 'gross' in str(col).lower()), None)
                year_col = next((col for col in df.columns if 'year' in str(col).lower()), None)
                title_col = next((col for col in df.columns if 'title' in str(col).lower()), None)
                
                # Clean and prepare the data as needed
                if year_col:
                    try:
                        df[year_col] = df[year_col].astype(str).str.extract(r'(\d{4})')[0].fillna(0).astype(float)
                    except:
                        print(f"Could not parse {year_col} column")
                
                if gross_col:
                    try:
                        def parse_gross(s):
                            if isinstance(s, str):
                                s = s.replace("$", "").replace(",", "")
                                try:
                                    return float(s)
                                except:
                                    return 0
                            return 0
                        df['GrossNum'] = df[gross_col].apply(parse_gross)
                    except:
                        print(f"Could not parse {gross_col} column")
                
                # Prepare a data summary for Gemini
                df_sample = df.head(5).to_string()
                data_summary = f"""
                Data from {url}:
                - Shape: {df.shape[0]} rows, {df.shape[1]} columns
                - Columns: {list(df.columns)}
                - Sample data:
                {df_sample}
                
                Questions to answer:
                {q_text}
                
                Please analyze this data and answer the questions. If JSON format is requested, 
                format your response as a JSON array of strings.
                """
                
                # Get Gemini's analysis of the data
                gemini_answer = call_gemini(data_summary)
                
                # Process specific visualizations if requested
                plot_data_uri = None
                if "scatterplot" in q_text.lower() or "plot" in q_text.lower() or "graph" in q_text.lower():
                    try:
                        # Try to identify what columns to use for the plot
                        x_col, y_col = None, None
                        
                        # Look for specific column mentions in the query
                        col_match = re.search(r'(?:plot|graph|chart|scatter)[^\w]+([\w\s]+)[^\w]+(?:vs|versus|against)[^\w]+([\w\s]+)', q_text.lower())
                        if col_match:
                            col1, col2 = col_match.groups()
                            x_col = next((col for col in df.columns if col1.strip() in str(col).lower()), None)
                            y_col = next((col for col in df.columns if col2.strip() in str(col).lower()), None)
                        
                        # If no specific columns mentioned, use best guess
                        if not (x_col and y_col):
                            if rank_col and 'GrossNum' in df.columns:
                                x_col, y_col = rank_col, 'GrossNum'
                            elif rank_col and year_col:
                                x_col, y_col = rank_col, year_col
                            elif len(numeric_cols) >= 2:
                                x_col, y_col = numeric_cols[0], numeric_cols[1]
                        
                        # Generate the plot if we have columns to use
                        if x_col and y_col:
                            plot_data_uri = generate_scatterplot(df, x_col, y_col)
                    except Exception as e:
                        print(f"Error generating plot: {e}")
                
                # Format the response
                if json_requested:
                    try:
                        # Try to extract JSON from Gemini's response
                        json_match = re.search(r'\[.*\]', gemini_answer, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            json_data = json.loads(json_str)
                            
                            # Add the plot if we generated one
                            if plot_data_uri:
                                # Try to find an appropriate place to put the plot
                                for i, item in enumerate(json_data):
                                    if "plot" in str(item).lower() or "scatter" in str(item).lower() or "graph" in str(item).lower():
                                        json_data[i] = plot_data_uri
                                        break
                                else:
                                    # If no appropriate place, append it
                                    json_data.append(plot_data_uri)
                                
                            all_responses.append(json.dumps(json_data))
                        else:
                            # Create our own JSON with Gemini's answer and the plot
                            result = [gemini_answer]
                            if plot_data_uri:
                                result.append(plot_data_uri)
                            all_responses.append(json.dumps(result))
                    except Exception as e:
                        print(f"Error formatting JSON response: {e}")
                        # Fallback
                        result = [gemini_answer]
                        if plot_data_uri:
                            result.append(plot_data_uri)
                        all_responses.append(json.dumps(result))
                else:
                    # Return text format
                    all_responses.append(gemini_answer)
                    if plot_data_uri:
                        all_responses.append(plot_data_uri)
            
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                # Fall back to Gemini for this URL
                gemini_prompt = f"Visit this URL: {url} and answer these questions: {q_text}"
                gemini_answer = call_gemini(gemini_prompt)
                
                if json_requested:
                    json_match = re.search(r'\[.*\]', gemini_answer, re.DOTALL)
                    if json_match:
                        all_responses.append(json_match.group())
                    else:
                        all_responses.append(json.dumps([gemini_answer]))
                else:
                    all_responses.append(gemini_answer)
        
        # Return all collected responses
        return all_responses
    
    except Exception as e:
        print(f"Error in analyze_questions: {e}")
        error_msg = f"Error analyzing questions: {str(e)}"
        if json_requested:
            return [json.dumps([error_msg])]
        else:
            return [error_msg]


# ===== Streamlit UI =====
def run_streamlit():
    st.set_page_config(page_title="Data Analyst Agent", layout="wide")
    st.title("📊 Data Analyst Agent - Streamlit + FastAPI")
    st.write("Upload one or more **questions.txt**, **CSV**, or **image file(s)** for analysis.")

    if 'loading' not in st.session_state:
        st.session_state.loading = False
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'scraped_df' not in st.session_state:
        st.session_state.scraped_df = None
    if 'img_llm_ocr_text' not in st.session_state:
        st.session_state.img_llm_ocr_text = None

    uploaded_files = st.file_uploader(
        "Upload one or more files (txt, csv, png, jpg, jpeg)",
        type=["txt", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    manual_question = st.text_area(
        "Or enter questions manually:",
        height=150,
        placeholder="Enter your questions here. Include URLs if you want to analyze data from websites."
    )

    if st.button("Analyze", type="primary"):
        if not uploaded_files and not manual_question:
            st.warning("Please upload at least one file or enter questions.")
        else:
            st.session_state.loading = True
            st.session_state.results = []
            st.session_state.scraped_df = None
            st.session_state.img_llm_ocr_text = None

            with st.spinner("🔄 Analyzing your request..."):
                time.sleep(0.5)
                try:
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            file_extension = uploaded_file.name.split(".")[-1].lower()
                            file_content = uploaded_file.read()

                            if file_extension in ["png", "jpg", "jpeg"]:
                                st.image(file_content, use_column_width=True, caption=f"Uploaded Image: {uploaded_file.name}")
                                img_base64 = base64.b64encode(file_content).decode('utf-8')
                                prompt = "Extract all textual content from this image as one plain text block."
                                ocr_text = call_gemini_image(prompt, img_base64, mime_type=uploaded_file.type)
                                st.session_state.img_llm_ocr_text = ocr_text
                                q_text = ocr_text
                            else:
                                q_text = file_content.decode("utf-8")

                            if manual_question:
                                q_text = f"{manual_question}\n\n{q_text}"

                            q_text = q_text.replace("\r", "").strip()
                            q_text = re.sub(r"\s+", " ", q_text)

                            answers = analyze_questions(q_text)
                            st.session_state.results.append((uploaded_file.name, answers))
                    else:
                        q_text = manual_question
                        q_text = q_text.replace("\r", "").strip()
                        q_text = re.sub(r"\s+", " ", q_text)
                        answers = analyze_questions(q_text)
                        st.session_state.results.append(("Manual Input", answers))

                except Exception as e:
                    st.session_state.results = [("Error", [f"Error analyzing your request: {str(e)}"])]

                st.session_state.loading = False

    if st.session_state.loading:
        st.info("⏳ Processing your request...")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)

    # Display results
    if st.session_state.results and not st.session_state.loading:
        st.success("✅ Analysis Complete!")

        for file_name, answers in st.session_state.results:
            st.subheader(f"Results for {file_name}:")
            
            for i, answer in enumerate(answers):
                if isinstance(answer, str) and answer.startswith("data:image/png;base64,"):
                    try:
                        base64_data = answer.split("base64,")[1]
                        image_data = base64.b64decode(base64_data)
                        st.image(image_data, use_column_width=True, caption="Generated Plot")
                    except Exception as e:
                        st.error(f"Could not display image: {e}")
                elif isinstance(answer, str) and (answer.startswith("[") and answer.endswith("]") or 
                                                  answer.startswith("{") and answer.endswith("}")):
                    # This might be a JSON string
                    try:
                        json_data = json.loads(answer)
                        st.json(json_data)
                        
                        # Display any images in the JSON
                        if isinstance(json_data, list):
                            for item in json_data:
                                if isinstance(item, str) and item.startswith("data:image/png;base64,"):
                                    try:
                                        base64_data = item.split("base64,")[1]
                                        image_data = base64.b64decode(base64_data)
                                        st.image(image_data, use_column_width=True)
                                    except Exception as e:
                                        st.error(f"Could not display image from JSON: {e}")
                    except:
                        # Not valid JSON, just display as text
                        st.markdown(answer)
                elif answer is None:
                    pass  # Skip None values
                else:
                    st.markdown(answer)

# ===== Run App =====
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

if __name__ == "__main__":
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=FASTAPI_PORT),
        daemon=True,
    ).start()
    run_streamlit()