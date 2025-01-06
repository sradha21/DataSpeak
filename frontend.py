import gradio as gr
from backend import read_data, eda_summary, generate_plot, process_query

def handle_file(file, file_type, task, plot_type=None, x=None, y=None):
    df = read_data(file.name, file_type)
    if task == 'EDA':
        summary = eda_summary(df)
        return f"Dataset Summary:\n\n{summary}"
    elif task == 'Visualization':
        if plot_type:
            plot = generate_plot(df, plot_type, x, y)
            return gr.Image(plot)
        else:
            return "Please select a plot type."
    else:
        return "Invalid task selected."

def query_dataset(query, file, file_type):
    df = read_data(file.name, file_type)
    if query.lower() in ["columns", "shape", "summary"]:
        if query.lower() == "columns":
            return f"Columns: {df.columns.tolist()}"
        elif query.lower() == "shape":
            return f"Shape: {df.shape}"
        elif query.lower() == "summary":
            return eda_summary(df)
    else:
        return process_query(query)

app = gr.TabbedInterface([file_input, query_input], ["EDA/Visualization", "Q&A"])
if __name__ == "__main__":
    app.launch()
