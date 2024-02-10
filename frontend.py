import tkinter as tk
from tkinter import filedialog
import pandas as pd

class DataFrameUploader(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DataFrame Uploader")
        self.geometry("400x200")
        
        self.file_path_label = tk.Label(self, text="No file chosen")
        self.file_path_label.pack(pady=10)
        
        self.upload_button = tk.Button(self, text="Upload DataFrame", command=self.upload_dataframe)
        self.upload_button.pack(pady=5)
        
    def upload_dataframe(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_label.config(text=file_path)
            try:
                df = pd.read_csv(file_path)
                # Do whatever you want with the DataFrame
                print(df.head())  # For demonstration, printing first few rows
            except Exception as e:
                self.file_path_label.config(text="Error loading file: " + str(e))

if __name__ == "__main__":
    app = DataFrameUploader()
    app.mainloop()