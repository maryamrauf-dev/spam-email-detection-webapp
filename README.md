# 📧 Spam vs Ham Email Classifier - AI Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Project Overview

This project is a high-performance **Spam Email Detection System** developed as part of my internship at **NovaXccelerate**. It leverages advanced Machine Learning techniques to differentiate between "Spam" and "Ham" (legitimate) messages with high precision.

The application features a modern, cyberpunk-inspired UI built with **Streamlit**, providing users with a seamless experience for both individual message analysis and batch processing via CSV uploads.

---

## ✨ Key Features

-   **⚡ Real-Time Analysis:** Instantly classify any email or SMS content as Spam or Ham using AI.
-   **📊 Batch Processing:** Upload a CSV file containing multiple messages and get a detailed classification report in seconds.
-   **🎨 Cyberpunk Aesthetic:** A premium, dark-themed UI with glow effects and responsive components.
-   **📜 Scan History:** Keep track of your recent scans with a convenient history sidebar.
-   **🔍 AI Confidence Scoring:** View the statistical confidence level of each prediction to understand the model's certainty.
-   **💾 Intelligent Initializer:** Automatically trains and optimizes the model on the first run using local datasets.

---

## 🏗️ Technical Architecture

### 🧠 The Model
The core intelligence is powered by a **Logistic Regression** classifier, optimized with:
-   **TF-IDF Vectorization:** Converts raw text into numerical features while emphasizing unique keywords.
-   **Balanced Class Weights:** Specifically tuned to handle the native imbalance between spam and ham messages.
-   **Scikit-Learn Pipeline:** Ensures consistency between training and inference phases.

### 🛠️ Tech Stack
-   **Frontend:** Streamlit (with Custom CSS/HTML Injection for premium UI)
-   **Backend:** Python 3.x
-   **Machine Learning:** Scikit-learn, Pandas, NumPy
-   **Data Storage:** Local CSV integration for efficient processing

---

## 📂 Project Structure

```text
├── data/
│   └── dataset/
│       └── dataset.csv      # The training data
├── app.py                   # Main Streamlit Application (UI & Logic)
├── model.py                 # Backend ML Logic (Training & Prediction)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── Spam Email Detection.ipynb # Exploration & Prototyping Notebook
```

---

## ⚙️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/spam-email-classifier.git
    cd "spam email detection internship project"
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

---

## 🚀 How to Use

1.  **Manual Check:** Paste your email content into the main text area and click **"VERIFY INTEGRITY"**.
2.  **Batch Check:** Use the sidebar to upload a CSV file (ensure it has a column named `Message` or `text`).
3.  **Review History:** Check the sidebar to see your previous scan results and their confidence levels.

---

## 🎓 Internship Project

This project was developed during my tenure as an AI/ML intern at **NovaXccelerate**. It demonstrates the practical application of Natural Language Processing (NLP), feature engineering, and modern web deployment strategies in a real-world scenario.

---

## 🤝 Contributing

Contributions are welcome! If you have any suggestions to improve the model accuracy or UI, feel free to fork the repo and submit a PR.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

> **Note:** This model is trained on general spam datasets. For specific enterprise needs, fine-tuning on domain-specific data is recommended.
