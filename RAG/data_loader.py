from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,CSVLoader,UnstructuredExcelLoader
from pathlib import Path
from typing import List,Any

def load_all_documents(data_dir:str)->List[Any]:
    data_path = Path(data_dir).resolve()
    documents = []

    # PDF Files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"Found {len(pdf_files)} PDF files to process")
    for pdf in pdf_files:
        print(f"\n Processing: {pdf.name}")
        try:
            pdf_loader = PyMuPDFLoader(str(pdf))
            pdf_loaded = pdf_loader.load()
            documents.extend(pdf_loaded)
        except Exception as e:
            print(f"Failed to load {pdf.name}: {e}")

    # TEXT Files
    text_files = list(data_path.glob('**/*.txt'))
    print(f"Found {len(text_files)} Text files to process")
    for txt_file in text_files:
        print(f"\n Processing: {txt_file.name}")
        try:
            text_loader = TextLoader(str(txt_file))
            text_loaded = text_loader.load()
            documents.extend(text_loaded)
        except Exception as e:
            print(f"Failed to load {txt_file.name}: {e}")

    # CSV File
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"Found {len(csv_files)} CSV files to process")
    for csv_file in csv_files:
        print(f"\n Processing: {csv_file.name}")
        try:
            csv_loader = CSVLoader(str(csv_file))
            csv_loaded = csv_loader.load()
            documents.extend(csv_loaded)
        except Exception as e:
            print(f"Failed to load {csv_file.name}: {e}")

    # EXCEL File
    excel_files = list(data_path.glob('**/*.xlsx'))
    print(f"Found {len(excel_files)} Excel files to process")
    for excel_file in excel_files:
        print(f"\nProcessing: {excel_file.name}")
        try:
            excel_loader = UnstructuredExcelLoader(str(excel_file))
            excel_loaded = excel_loader.load()
            documents.extend(excel_loaded)
        except Exception as e:
            print(f"Failed to load {excel_file.name}: {e}")
    return documents
