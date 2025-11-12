# Telematics Automated Customer Report Generator


## 📌 Overview
A Python application for automating telematics data processing:
- Downloads vehicle data via API
- Processes MATLAB-formatted files
- Generates Excel reports for customers

## 📂 Project Structure

<pre>
python_ev_code/
├── src/
│   ├── api_client.py
│   ├── file_downloader.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── .gitignore/
├── .env/  # Secrets file (IGNORED by Git)
├── .env.example        # Template (safe to commit)
├── requirements.txt
└── README.md

</pre>


## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git (optional)

### Installation
```bash
# Clone repository (if using Git)
git clone https://your-repository-url.git
cd python_ev_code

# Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

### 🛠 Usage
```bash
# Run main processing pipeline
python src/main.py --customer WestCoast

# For help
python src/main.py --help
```

### 📝 Version History

| Version | Date       | Description                          |
|---------|------------|--------------------------------------|
| 0.1.0   | 2025-09-07 | Initial release with core features   |




