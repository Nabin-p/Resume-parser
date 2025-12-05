# Resume/CV Parser

A web application that extracts key information from resumes (PDF or image) using NER (Named Entity Recognition) with a fine-tuned Hugging Face model.

---

## 🚀 Features

- Extracts **Name, Email, Phone, Location, Skills, Education, Experience** from resumes.  
- Supports **PDF and image files**.  
- Uses a fine-tuned **BERT-based NER model**.  
- Provides instant results with a **preview of the uploaded resume**.  
- Built with **Streamlit** for a simple, interactive interface.

---

## 📁 Dataset

- Fine-tuned on Kaggle Resume NER dataset: `resume-ner-training-data`  
- Custom model hosted on Hugging Face: [Nabinp10/resume-ner-output](https://huggingface.co/Nabinp10/resume-ner-output)

---

## 🛠 Installation

Clone this repository:

```bash
git clone https://github.com/nabin-p/Resume-parser.git
cd resume-parser
````

Create a virtual environment and activate it:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your **Poppler path** for PDF processing (Windows):

```python
poppler_path = r"C:\path\to\poppler\bin"
```

---

## ⚡ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Upload a **PDF or image resume** and see the extracted information instantly.

---

## 🧠 How It Works

1. Upload a resume (**PDF or image**).
2. Convert PDF pages to text using **PyMuPDF (fitz)**.
3. Extract entities using **Hugging Face NER pipeline** with the fine-tuned **BERT model**.
4. Display results for **Name, Email, Phone, Location, Skills, Education, Experience**.

---

## 💻 Tech Stack

* Python 3.x
* Streamlit
* PyMuPDF (fitz)
* PDF2Image
* Pillow (PIL)
* Hugging Face Transformers & Datasets
* Regex for emails, phones, and experience parsing

---

## 🔗 Model & Dataset

* **Hugging Face Model:** [Nabinp10/resume-ner-output](https://huggingface.co/Nabinp10/resume-ner-output)
* **Dataset:** Kaggle Resume NER Dataset

---

## ⚠️ Notes

* Make sure you have enough disk space to load the model (~400 MB).
* The current model is fine-tuned on Kaggle dataset and may not be perfect for all resume formats.

