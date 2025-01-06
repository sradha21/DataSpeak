import pandas as pd
from transformers import pipeline
import json
import sqlite3
import PyPDF2
import openpyxl

# Initialize the GPT model
gpt_generator = pipeline('text-generation', model='gpt2')

def read_data(file_path, file_type):
    """
    Reads data from various file types and returns it in a suitable format.
    """
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'excel':
        return pd.read_excel(file_path)
    elif file_type == 'json':
        with open(file_path, 'r') as file:
            return json.load(file)
    elif file_type == 'pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
    elif file_type == 'sql':
        conn = sqlite3.connect(file_path)
        query = "SELECT * FROM tablename"  # Customize this query
        return pd.read_sql_query(query, conn)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def process_query(query):
    """
    Processes a natural language query using a pre-trained GPT model.
    """
    response = gpt_generator(query, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

if __name__ == "__main__":
    # Add a small test block here
    print("Running backend.py")
