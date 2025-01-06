import pandas as pd
from transformers import pipeline
import json
import sqlite3
import PyPDF2
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Initialize GPT model for Q&A
gpt_generator = pipeline('text-generation', model='gpt2')

def process_query(query):
    response = gpt_generator(query, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

def read_data(file_path, file_type):
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'excel':
        return pd.read_excel(file_path)
    elif file_type == 'json':
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    elif file_type == 'pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
    elif file_type == 'sql':
        conn = sqlite3.connect(file_path)
        query = "SELECT * FROM tablename"
        return pd.read_sql_query(query, conn)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def eda_summary(df):
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "summary_statistics": df.describe(include='all').to_dict(),
    }
    return summary

def generate_plot(df, plot_type, x=None, y=None):
    plt.figure(figsize=(10, 6))
    if plot_type == 'histogram':
        df[x].hist()
    elif plot_type == 'scatter' and x and y:
        plt.scatter(df[x], df[y])
    elif plot_type == 'heatmap':
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    else:
        return "Invalid plot type or missing columns."
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

if __name__ == "__main__":
    print("Backend ready.")
