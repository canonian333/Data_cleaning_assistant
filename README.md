# 🧹 Interactive Data Cleaning Assistant

> Upload, Analyze, and Clean Your Data with Ease – Now Enhanced with AI-Powered Insights via RAG!

## 🔍 Overview

Data cleaning is one of the most time-consuming yet critical parts of any data science or analytics workflow. Our **Interactive Data Cleaning Assistant** streamlines this process using an intuitive web interface built with **Streamlit** and augmented with **Retrieval Augmented Generation (RAG)** for intelligent metadata querying.

### ✨ Key Features

#### 📊 Automated Data Profiling
- Generates an instant metadata summary of your uploaded CSV:
  - Data types
  - Missing values
  - Unique value counts
  - Descriptive statistics: mean, median, skewness, kurtosis

#### 📈 Visual Insights
- Built-in data visualizations for numerical columns:
  - Histograms
  - Box plots (highlighting potential outliers)

#### 🧠 Smart Cleaning Recommendations
- Suggests fixes based on common issues:
  - Missing value strategies (drop, fill, flag)
  - Outlier detection using IQR and Z-score methods
  - Type inconsistency identification (e.g., numbers stored as text)
  - Cardinality warnings for categorical variables

#### 🤖 Ask Your Data with RAG
- Integrated **RAG chatbot** powered by **ChromaDB** and **Sentence Transformers**
- Ask questions like:
  - “What is the mean of Age?”
  - “Which columns have missing data?”
  - “What are the outcome variables?”
- Get context-aware answers grounded in your dataset’s metadata.

---

## 🛠️ Built With

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 🚀 Getting Started

### 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/data-cleaning-assistant.git
cd data-cleaning-assistant
pip install -r requirements.txt
