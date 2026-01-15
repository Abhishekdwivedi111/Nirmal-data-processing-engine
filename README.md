# Nirmal – Intelligent Data Processing Engine

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

> **A production-ready, modular Python framework for automated data cleaning, validation, and preprocessing with YAML-driven configuration and interactive web interface.**

---

## 🎯 Overview

**Nirmal** is an enterprise-grade data processing engine that handles real-world data quality challenges. It provides a flexible, configuration-driven approach to building data cleaning pipelines that work with **any dataset** without hardcoded column names.

### Why "Nirmal"?

*Nirmal* means "clean" or "pure" in Sanskrit, reflecting the engine's core purpose: transforming messy, real-world data into clean, analysis-ready datasets.

---

## ✨ Key Features

- 🔧 **Modular Architecture**: Clean separation of concerns with dedicated modules
- 📝 **YAML Configuration**: Define pipelines through simple YAML files
- 📄 **Multiple File Formats**: Supports CSV, Excel, JSON, Parquet, and YAML data files
- 🤖 **Auto-Detection**: Automatically detects data types and applies appropriate strategies
- 📊 **Dual Interface**: Web UI (Streamlit) and CLI for automation
- 📈 **Processing Summary**: Real-time statistics and detailed validation reports
- 🔍 **Comprehensive Validation**: Data type, range, null, and custom validators
- 🎯 **Outlier Handling**: Configurable IQR/Z-score methods with capping or removal
- 📝 **Structured Logging**: Comprehensive logging for tracking execution
- 💾 **Flexible Output**: Download processed data in CSV, Excel, JSON, or YAML formats

---

---
## 👁️👁️ Live Demo
Check out the live version of this app here:

[![Live App](https://img.shields.io/badge/Streamlit-Live-brightgreen)](https://nirmal-data-processing-engine-h9vgh8ckuvbdxjp4hpdsoj.streamlit.app/)

Try the app online to process your data in real-time and download clean results!

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nirmal.git
cd nirmal

# Install dependencies
pip install -r requirements.txt
```

### Web Interface (Recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` and start processing data interactively!

### Command-Line Interface

```bash
python main.py --config config/default_config.yaml --input data/input.csv --output data/output.csv
```

---

## 📋 Configuration

### Generic Configuration (Works with ANY Dataset)

The `config/default_config.yaml` works with **any dataset** without hardcoded column names:

```yaml
pipeline:
  cleaning:
    enabled: true
  validation:
    enabled: true
  preprocessing:
    enabled: true

cleaning:
  remove_duplicates:
    enabled: true
    subset: null  # Auto-excludes ID columns
  
  handle_missing:
    enabled: true
    strategy: "fill"  # Auto-detects: median for numeric, mode for text
  
  outliers:
    enabled: true
    method: "iqr"
    cap: true  # Cap outliers instead of removing rows
    columns: []  # Empty = auto-detect all numeric columns
```

**The engine automatically skips missing columns**, so you can safely use a config even if some columns don't exist in your data.

---

## 📊 Processing Summary

After processing, the engine provides comprehensive statistics:

```
Processing Summary:
✔ 190,769 missing values handled
✔ 541 duplicates removed
✔ 11 numeric columns validated
✔ 15 text columns cleaned
```

---

## 📁 Project Structure

```
nirmal/
├── nirmal/                    # Core engine package
│   ├── engine.py              # Main orchestrator
│   ├── cleaners.py            # Data cleaning operations
│   ├── validators.py          # Data validation operations
│   ├── preprocessors.py       # Feature engineering
│   ├── config_loader.py       # YAML configuration parser
│   └── logger_setup.py        # Structured logging
│
├── config/                    # Configuration files
│   ├── default_config.yaml    # ⭐ Generic config (works with ANY dataset)
│   ├── example_config.yaml    # Template with all options
│   └── airbnb_config.yaml     # Example: Airbnb-specific
│
├── app.py                     # Streamlit web interface
├── main.py                    # CLI entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 💻 Usage Examples

### Basic Usage

```python
from nirmal.engine import DataProcessingEngine

# Initialize engine
engine = DataProcessingEngine(config_path='config/default_config.yaml')

# Process data
df_processed = engine.process('data/input.csv', 'data/output.csv')

# Get results
summary = engine.get_processing_summary()
print(f"Processed {summary['final_rows']} rows")
```

### Custom Configuration

```python
config = {
    'pipeline': {'cleaning': {'enabled': True}},
    'cleaning': {
        'remove_duplicates': {'enabled': True},
        'handle_missing': {'enabled': True, 'strategy': 'fill'},
        'outliers': {'enabled': True, 'method': 'iqr', 'cap': True}
    }
}

engine = DataProcessingEngine(config_dict=config)
df = engine.process('data/input.csv')
```

---

## 🧪 Testing

```bash
# Generate sample data
python create_sample_data.py

# Run tests
python test_nirmal.py
```

---

## 🔧 Technical Stack

- **Python 3.7+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **PyYAML**: Configuration parsing
- **Streamlit**: Web interface
- **Scikit-learn**: Preprocessing utilities
- **PyArrow**: Parquet file support
- **OpenPyXL**: Excel file support

## 📄 Supported File Formats

### Input Formats
- **CSV** (.csv) - Comma-separated values
- **Excel** (.xlsx, .xls) - Microsoft Excel files
- **JSON** (.json) - JavaScript Object Notation
- **Parquet** (.parquet) - Columnar storage format
- **YAML** (.yaml, .yml) - YAML data files (list of records or structured data)

### Output Formats
- **CSV** (.csv)
- **Excel** (.xlsx)
- **JSON** (.json)
- **YAML** (.yaml, .yml) - With metadata

---

## 💡 Use Cases

- **Data Science**: Preprocess datasets before model training
- **ETL Pipelines**: Clean data from multiple sources
- **Data Quality Monitoring**: Run validation checks on incoming data
- **Research & Analysis**: Quick data cleaning for exploratory analysis

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

For questions, issues, or feature requests, please open an issue in the repository.

---

**Built with attention to detail, designed for production use.**
