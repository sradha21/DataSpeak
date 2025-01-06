# backend.py

import pandas as pd
from transformers import pipeline
import json
import sqlite3  # For SQL database interactions
import PyPDF2  # For PDF file reading
import openpyxl  # For Excel file reading

# Initialize a GPT model using the Hugging Face transformers pipeline
gpt_generator = pipeline('text-generation', model='gpt2')

def process_query(query):
    """
    Processes a natural language query using a pre-trained GPT model.
    """
    response = gpt_generator(query, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

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
        conn = sqlite3.connect(file_path)  # Assuming SQLite for simplicity
        query = "SELECT * FROM tablename"  # You need to customize this query
        return pd.read_sql_query(query, conn)

# Example usage
if __name__ == "__main__":
    query = "Tell me about AI in healthcare."
    print("Query: ", query)
    print("Response: ", process_query(query))
