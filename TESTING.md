# Testing Guide for Nirmal Data Processing Engine

This guide explains how to test and run the Nirmal project.

## Prerequisites

1. Python 3.7 or higher installed
2. All dependencies installed (run `pip install -r requirements.txt`)

## Quick Test

### Step 1: Generate Sample Data
```bash
python create_sample_data.py
```

This creates `data/sample_input.csv` with various data quality issues:
- Duplicate rows
- Missing values
- Outliers
- Inconsistent text formatting

### Step 2: Run the Test Script
```bash
python test_nirmal.py
```

This will:
- Load the sample data
- Process it through the pipeline
- Display before/after statistics
- Show validation results
- Save processed data to `data/sample_output.csv`

## Command Line Usage

### Basic Usage
```bash
python main.py --config config/test_config.yaml --input data/sample_input.csv --output data/output.csv
```

### With Custom Log Level
```bash
python main.py --config config/test_config.yaml --input data/sample_input.csv --output data/output.csv --log-level DEBUG
```

### Parameters
- `--config`: Path to YAML configuration file (required)
- `--input`: Path to input data file (required)
- `--output`: Path to output data file (optional)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)

## Testing with Your Own Data

1. **Prepare your data file** (CSV, Excel, JSON, or Parquet format)
2. **Create or modify a configuration file** (see `config/example_config.yaml`)
3. **Run the engine:**
   ```bash
   python main.py --config config/your_config.yaml --input data/your_data.csv --output data/processed_output.csv
   ```

## Expected Output

The engine will:
1. Load your data file
2. Apply cleaning operations (remove duplicates, handle missing values, etc.)
3. Validate data quality
4. Apply preprocessing transformations
5. Save processed data to output file
6. Generate logs in the `logs/` directory

## Checking Results

- **Console Output**: Processing summary and validation results
- **Log Files**: Detailed logs in `logs/nirmal_YYYYMMDD_HHMMSS.log`
- **Output File**: Processed data saved to specified output path

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Ensure input file path is correct
   - Check configuration file path

2. **Import Errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

3. **Validation Errors**
   - Check configuration file structure
   - Review log files for detailed error messages
   - Set `strict_validation: false` in config for non-critical validation failures

4. **Memory Issues**
   - Process data in chunks for very large files
   - Use appropriate data types

## Verification Checklist

- [ ] Dependencies installed successfully
- [ ] Sample data generated
- [ ] Test script runs without errors
- [ ] Output file created
- [ ] Log files generated
- [ ] Data processing visible in console output
