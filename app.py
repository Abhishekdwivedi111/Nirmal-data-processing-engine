"""
Streamlit web application for Nirmal Data Processing Engine.
"""

import streamlit as st
import pandas as pd
import io
import yaml
from pathlib import Path
import tempfile
import os

from nirmal.engine import DataProcessingEngine
from nirmal.logger_setup import setup_logger

# Page configuration
st.set_page_config(
    page_title="Nirmal - Data Processing Engine",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = setup_logger(log_level="INFO")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    /* Enhance file uploader styling for better drag and drop visibility */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #1f77b4 !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        background-color: #f0f8ff !important;
        min-height: 100px !important;
        max-height: 200px !important;
        overflow: auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #0d6efd !important;
        background-color: #e7f3ff !important;
        border-width: 4px !important;
    }
    /* Make the upload button more visible */
    div[data-testid="stFileUploader"] button {
        background-color: #1f77b4 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: bold !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #0d6efd !important;
    }
    /* Style the drag area text */
    div[data-testid="stFileUploader"] p {
        font-size: 1.1rem !important;
        color: #1f77b4 !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)

def is_enabled(config, default=False):
    """
    Safely checks if a config section is enabled.
    Supports:
    - bool → true / false
    - dict → { enabled: true }
    - None → default
    """
    if isinstance(config, bool):
        return config
    if isinstance(config, dict):
        return config.get("enabled", default)
    return default


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame to Arrow-compatible format for Streamlit display.
    Converts object columns to string to avoid ArrowInvalid errors.
    """
    df_display = df.copy()
    
    # Convert all object columns to string to make them Arrow-compatible
    for col in df_display.columns:
        if df_display[col].dtype == 'object':
            # Convert to string, handling NaN values
            df_display[col] = df_display[col].astype(str).replace('nan', pd.NA)
    
    # Also handle any mixed-type columns that might cause issues
    # Convert any remaining problematic types
    for col in df_display.columns:
        try:
            # Try to convert to a safe type
            if df_display[col].dtype == 'object':
                # Already handled above, but ensure it's string
                df_display[col] = df_display[col].astype(str)
        except Exception:
            # If conversion fails, keep as is
            pass
    
    return df_display


def load_default_config():
    """Load default configuration."""
    # Try generic default config first, then fallback to test_config
    config_paths = [
        Path("config/default_config.yaml"),
        Path("config/test_config.yaml")
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    return None


def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temporary directory."""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_pipeline_steps(config_dict):
    """Extract enabled pipeline steps from configuration."""
    steps = []
    
    if not config_dict:
        return steps
    
    pipeline_config = config_dict.get('pipeline', {})
    cleaning_config = config_dict.get('cleaning', {})
    validation_config = config_dict.get('validation', {})
    preprocessing_config = config_dict.get('preprocessing', {})
    
    # Cleaning steps
    if pipeline_config.get('cleaning', {}).get('enabled', True):
        if is_enabled(cleaning_config.get('remove_duplicates')):
            steps.append("Remove duplicates")
        if cleaning_config.get('handle_missing', {}).get('enabled', False):
            steps.append("Handle missing values")
        if cleaning_config.get('outliers', {}).get('enabled', False) or cleaning_config.get('remove_outliers', {}).get('enabled', False):
            steps.append("Handle outliers")
        if cleaning_config.get('standardize_text', {}).get('enabled', False):
            steps.append("Clean text columns")
    
    # Validation steps
    if pipeline_config.get('validation', {}).get('enabled', True):
        if validation_config.get('data_types', {}).get('enabled', False):
            steps.append("Validate data types")
        if validation_config.get('value_ranges', {}).get('enabled', False):
            steps.append("Validate value ranges")
        if validation_config.get('not_null', {}).get('enabled', False):
            steps.append("Validate not-null constraints")
        if validation_config.get('unique', {}).get('enabled', False):
            steps.append("Validate unique constraints")
    
    # Preprocessing steps
    if pipeline_config.get('preprocessing', {}).get('enabled', True):
        if preprocessing_config.get('normalize', {}).get('enabled', False):
            steps.append("Normalize features")
        if preprocessing_config.get('encode_categorical', {}).get('enabled', False):
            steps.append("Encode categorical features")
        if preprocessing_config.get('feature_selection', {}).get('enabled', False):
            steps.append("Feature selection")
    
    return steps


def main():
    """Main Streamlit application."""
    
    # Header with subtitle
    st.markdown('<div class="main-header">🔧 Nirmal – Intelligent Data Processing Engine</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.1rem; margin-top: -1rem; margin-bottom: 2rem;">Configurable, YAML-driven data cleaning & validation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    An intelligent, modular Python engine for automated data cleaning, validation, and preprocessing.
    Upload your data file and configure the processing pipeline using the options below.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Configuration mode
        config_mode = st.radio(
            "Configuration Mode",
            ["Use Default Config", "Upload Config File", "Quick Settings"],
            index=0
        )
        
        config_dict = None
        config_file_path = None
        
        if config_mode == "Use Default Config":
            default_config = load_default_config()
            if default_config:
                config_dict = default_config
                st.success("Using default configuration")
            else:
                st.warning("Default config not found. Using minimal config.")
                config_dict = {
                    'pipeline': {
                        'cleaning': {'enabled': True},
                        'validation': {'enabled': True},
                        'preprocessing': {'enabled': True},
                        'strict_validation': False
                    },
                    'cleaning': {
                        'remove_duplicates': {'enabled': True, 'keep': 'first'},
                        'handle_missing': {'enabled': True, 'strategy': 'fill', 'fill_value': 0}
                    },
                    'validation': {
                        'data_types': {'enabled': False},
                        'value_ranges': {'enabled': False}
                    },
                    'preprocessing': {}
                }
        
        elif config_mode == "Upload Config File":
            uploaded_config = st.file_uploader(
                "Upload YAML Config File",
                type=['yaml', 'yml'],
                help="Upload your YAML configuration file"
            )
            if uploaded_config:
                try:
                    # Reset file pointer to beginning
                    uploaded_config.seek(0)
                    config_content = uploaded_config.read().decode('utf-8')
                    # Reset again after reading
                    uploaded_config.seek(0)
                    
                    # Validate YAML content
                    if not config_content.strip():
                        st.error("Error: Configuration file is empty")
                        config_dict = None
                    else:
                        config_dict = yaml.safe_load(config_content)
                        
                        # Validate that config_dict is not None and has required structure
                        if config_dict is None:
                            st.error("Error: Configuration file is empty or invalid")
                            config_dict = None
                        elif not isinstance(config_dict, dict):
                            st.error("Error: Configuration must be a YAML dictionary/mapping")
                            config_dict = None
                        else:
                            st.success("Configuration loaded successfully!")
                            with st.expander("View Config"):
                                st.code(config_content, language='yaml')
                except yaml.YAMLError as e:
                    st.error(f"YAML parsing error: {e}")
                    config_dict = None
                except UnicodeDecodeError as e:
                    st.error(f"Encoding error: File must be UTF-8 encoded. {e}")
                    config_dict = None
                except Exception as e:
                    st.error(f"Error loading config: {e}")
                    logger.exception("Error loading YAML config file")
                    config_dict = None
            else:
                config_dict = None
        
        elif config_mode == "Quick Settings":
            st.subheader("Quick Settings")
            
            # Cleaning options
            st.markdown("### Cleaning")
            remove_duplicates = st.checkbox("Remove Duplicates", value=True)
            handle_missing = st.checkbox("Handle Missing Values", value=True)
            missing_strategy = st.selectbox(
                "Missing Values Strategy",
                ["fill", "drop", "forward_fill", "backward_fill"],
                disabled=not handle_missing
            )
            
            # Validation options
            st.markdown("### Validation")
            enable_validation = st.checkbox("Enable Validation", value=True)
            strict_validation = st.checkbox("Strict Validation", value=False, disabled=not enable_validation)
            
            # Preprocessing options
            st.markdown("### Preprocessing")
            enable_preprocessing = st.checkbox("Enable Preprocessing", value=False)
            
            # Build config dict
            config_dict = {
                'pipeline': {
                    'cleaning': {'enabled': True},
                    'validation': {'enabled': enable_validation},
                    'preprocessing': {'enabled': enable_preprocessing},
                    'strict_validation': strict_validation
                },
                'cleaning': {
                    'remove_duplicates': {'enabled': remove_duplicates, 'keep': 'first'},
                    'handle_missing': {
                        'enabled': handle_missing,
                        'strategy': missing_strategy,
                        'fill_value': 0 if missing_strategy == 'fill' else None
                    }
                },
                'validation': {
                    'data_types': {'enabled': False},
                    'value_ranges': {'enabled': False}
                },
                'preprocessing': {}
            }
        
        # Pipeline Steps Preview
        if config_dict:
            st.markdown("---")
            st.markdown("### 🔄 Pipeline Steps")
            steps = get_pipeline_steps(config_dict)
            if steps:
                for step in steps:
                    st.markdown(f"✔ {step}")
            else:
                st.info("Configure pipeline steps above")
        
        st.markdown("---")
        st.markdown("### 📖 Help")
        st.info("""
        **Supported Formats:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - JSON (.json)
        - Parquet (.parquet)
        - YAML (.yaml, .yml)
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Data")
        
        # Instructions for drag and drop
        st.info("💡 **Drag and drop** your file below or **click to browse**. Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet, YAML (.yaml, .yml)")
        
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'yaml', 'yml'],
            help="Drag and drop your file here or click to browse. Supported formats: CSV, Excel, JSON, Parquet, YAML",
            accept_multiple_files=False
        )
        
        if uploaded_file is None:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; border: 2px dashed #ccc; border-radius: 10px; background-color: #f8f9fa; margin-top: 1rem;">
                <p style="color: #6c757d; font-size: 1.1rem;">📁 No file uploaded yet</p>
                <p style="color: #6c757d;">Drag and drop a file above or click to browse</p>
            </div>
            """, unsafe_allow_html=True)
        
        if uploaded_file:
            # Instant upload feedback - show file info first
            file_size_mb = uploaded_file.size / (1024 * 1024)
            file_size_str = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{uploaded_file.size / 1024:.2f} KB"
            
            # Show file name and size immediately
            st.success("✅ File uploaded successfully!")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("File", uploaded_file.name)
            with col_info2:
                st.metric("Size", file_size_str)
            
            # Preview uploaded data
            df_preview = None
            try:
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                # Read file based on extension
                if file_extension == '.csv':
                    df_preview = pd.read_csv(uploaded_file)
                elif file_extension in ['.xlsx', '.xls']:
                    df_preview = pd.read_excel(uploaded_file)
                elif file_extension == '.json':
                    df_preview = pd.read_json(uploaded_file)
                elif file_extension == '.parquet':
                    df_preview = pd.read_parquet(uploaded_file)
                elif file_extension in ['.yaml', '.yml']:
                    # Read YAML file as data
                    uploaded_file.seek(0)
                    yaml_content = uploaded_file.read().decode('utf-8')
                    yaml_data = yaml.safe_load(yaml_content)
                    
                    # Convert YAML to DataFrame
                    # Use dtype='object' to prevent type inference issues
                    if isinstance(yaml_data, list):
                        df_preview = pd.DataFrame(yaml_data, dtype='object')
                    elif isinstance(yaml_data, dict):
                        if 'data' in yaml_data and isinstance(yaml_data['data'], list):
                            df_preview = pd.DataFrame(yaml_data['data'], dtype='object')
                        elif 'records' in yaml_data and isinstance(yaml_data['records'], list):
                            df_preview = pd.DataFrame(yaml_data['records'], dtype='object')
                        else:
                            df_preview = pd.DataFrame([yaml_data], dtype='object')
                    else:
                        st.error("YAML file must contain a list or dictionary")
                        st.session_state['file_uploaded'] = False
                        st.stop()
                else:
                    st.error("Unsupported file format")
                    st.session_state['file_uploaded'] = False
                    st.stop()
                
                # Reset file pointer for later use (needed for processing)
                uploaded_file.seek(0)
                
                # Show dimensions after successful read
                col_info3 = st.columns(1)[0]
                with col_info3:
                    st.metric("Dimensions", f"{df_preview.shape[0]:,} rows × {df_preview.shape[1]} columns")
                
                # Store preview data in session state for Process button
                st.session_state['df_preview'] = df_preview
                st.session_state['file_uploaded'] = True
                
                st.subheader("📊 Data Preview")
                
                with st.expander("View First 10 Rows", expanded=False):
                    # Make DataFrame Arrow-compatible before display
                    df_display = make_arrow_compatible(df_preview.head(10))
                    st.dataframe(df_display, use_container_width=True, height=300)
                
                with st.expander("Data Statistics"):
                    # Describe() returns numeric stats, should be fine, but make compatible just in case
                    df_stats = df_preview.describe()
                    df_stats_display = make_arrow_compatible(df_stats)
                    st.dataframe(df_stats_display, use_container_width=True)
                
                with st.expander("Data Info"):
                    info_buffer = io.StringIO()
                    df_preview.info(buf=info_buffer)
                    st.text(info_buffer.getvalue())
                
                # Data quality summary
                st.subheader("📈 Data Quality Summary")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    missing_count = df_preview.isnull().sum().sum()
                    st.metric("Missing Values", missing_count)
                
                with col_b:
                    duplicate_count = df_preview.duplicated().sum()
                    st.metric("Duplicate Rows", duplicate_count)
                
                with col_c:
                    numeric_cols = len(df_preview.select_dtypes(include=['number']).columns)
                    st.metric("Numeric Columns", numeric_cols)
                
                with col_d:
                    categorical_cols = len(df_preview.select_dtypes(include=['object']).columns)
                    st.metric("Text Columns", categorical_cols)
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                # Handle specific error types without referencing pd in exception handler
                if "pd" in error_msg.lower() or "pandas" in error_msg.lower() or "cannot access local variable" in error_msg.lower():
                    st.error("❌ Error: Pandas library issue detected.")
                    st.error("Please ensure pandas is properly installed: `pip install pandas`")
                    st.info("If the error persists, try restarting the Streamlit app.")
                elif "empty" in error_msg.lower() or "no columns" in error_msg.lower() or "EmptyDataError" in error_type:
                    st.error("❌ Error: The file appears to be empty or has no data.")
                elif "decode" in error_msg.lower() or "encoding" in error_msg.lower():
                    st.error("❌ Error: File encoding issue. Try saving the file as UTF-8.")
                elif "ImportError" in error_type:
                    st.error(f"❌ Import Error: {error_msg}")
                    st.error("Pandas library is not properly installed. Please run: pip install pandas")
                else:
                    st.error(f"❌ Error reading file ({error_type}): {error_msg}")
                
                st.info("💡 **Troubleshooting tips:**\n- Ensure the file is not corrupted\n- Check that the file format matches the extension\n- For CSV files, ensure proper encoding (UTF-8)\n- For large files, try processing in smaller chunks")
                
                logger.exception(f"File reading error: {error_type} - {error_msg}")
                st.session_state['file_uploaded'] = False
                if 'df_preview' in st.session_state:
                    del st.session_state['df_preview']
        else:
            st.session_state['file_uploaded'] = False
    
    with col2:
        st.header("🚀 Process")
        
        # Check if file and config are ready
        file_ready = st.session_state.get('file_uploaded', False)
        config_ready = config_dict is not None
        
        # Process button - enabled only when file and config are ready
        if file_ready and config_ready:
            if st.button("▶️ Process Data", type="primary", use_container_width=True):
                with st.spinner("Processing data..."):
                    try:
                        # Create temporary directory
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save uploaded file
                            input_path = save_uploaded_file(uploaded_file, temp_dir)
                            
                            # Initialize engine
                            # Debug: Show config being used
                            if config_dict:
                                st.write("🔍 **Debug Info:**")
                                with st.expander("View Config Being Used"):
                                    st.json(config_dict)
                            
                            engine = DataProcessingEngine(config_dict=config_dict)
                            
                            # Process data with progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Initializing engine...")
                            progress_bar.progress(10)
                            
                            # Process data
                            output_path = os.path.join(temp_dir, f"processed_{uploaded_file.name}")
                            
                            status_text.text("Cleaning data...")
                            progress_bar.progress(30)
                            
                            df_processed = engine.process(input_path, output_path)
                            
                            status_text.text("Validating results...")
                            progress_bar.progress(80)
                            
                            # Store results in session state
                            st.session_state['processed_df'] = df_processed
                            st.session_state['processed_file_name'] = f"processed_{uploaded_file.name}"
                            st.session_state['validation_results'] = engine.get_validation_results()
                            st.session_state['processing_summary'] = engine.get_processing_summary()
                            
                            progress_bar.progress(100)
                            status_text.text("Complete!")
                            
                            st.success("✅ Processing completed successfully!")
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                    except Exception as e:
                        st.error(f"❌ Error during processing: {e}")
                        logger.exception("Processing error")
        else:
            # Disabled button state
            button_disabled = True
            if not file_ready:
                st.info("📤 Upload a data file to enable processing")
            elif not config_ready:
                st.info("⚙️ Configure settings to enable processing")
            
            st.button("▶️ Process Data", disabled=True, use_container_width=True)
            
            # Show what's needed
            if not file_ready and not config_ready:
                st.markdown("""
                <div style="padding: 1rem; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;">
                    <strong>Ready to process?</strong><br>
                    • Upload a data file<br>
                    • Configure processing settings
                </div>
                """, unsafe_allow_html=True)
        
        # Display results if available
        if 'processed_df' in st.session_state:
                st.markdown("---")
                st.header("📥 Download Results")
                
                processed_df = st.session_state['processed_df']
                
                # Download section with improved layout
                col_dl1, col_dl2 = st.columns([2, 1])
                
                with col_dl1:
                    output_format = st.selectbox("Download Format", ["CSV", "Excel", "JSON", "YAML"], help="Choose the format for your processed data")
                
                with col_dl2:
                    # View Logs toggle
                    view_logs = st.checkbox("📋 View Logs", help="View processing logs")
                
                # Show logs if requested
                if view_logs:
                    with st.expander("📋 Processing Logs", expanded=True):
                        log_dir = Path("logs")
                        if log_dir.exists():
                            log_files = sorted(log_dir.glob("nirmal_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
                            if log_files:
                                latest_log = log_files[0]
                                try:
                                    with open(latest_log, 'r', encoding='utf-8') as f:
                                        log_content = f.read()
                                    st.text_area("Latest Log File", log_content, height=300, label_visibility="collapsed")
                                    st.caption(f"Log file: {latest_log.name}")
                                except Exception as e:
                                    st.error(f"Error reading log file: {e}")
                            else:
                                st.info("No log files found")
                        else:
                            st.info("Logs directory not found")
                
                st.markdown("---")
                
                # Ensure DataFrame is clean for downloads (handle NaN and mixed types)
                df_for_download = processed_df.copy()
                # Replace NaN with None for better serialization in JSON/YAML
                df_for_download = df_for_download.where(pd.notnull(df_for_download), None)
                
                if output_format == "CSV":
                    csv = df_for_download.to_csv(index=False, na_rep='')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=st.session_state['processed_file_name'].replace('.xlsx', '.csv').replace('.xls', '.csv').replace('.json', '.csv').replace('.parquet', '.csv').replace('.yaml', '.csv').replace('.yml', '.csv'),
                        mime="text/csv",
                        use_container_width=True
                    )
                elif output_format == "Excel":
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_for_download.to_excel(writer, index=False, sheet_name='Processed Data', na_rep='')
                    st.download_button(
                        label="📥 Download Excel",
                        data=output.getvalue(),
                        file_name=st.session_state['processed_file_name'].replace('.csv', '.xlsx').replace('.json', '.xlsx').replace('.xls', '.xlsx').replace('.parquet', '.xlsx').replace('.yaml', '.xlsx').replace('.yml', '.xlsx'),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                elif output_format == "JSON":
                    # Convert NaN to None for JSON serialization
                    json_str = df_for_download.to_json(orient='records', indent=2, date_format='iso')
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=st.session_state['processed_file_name'].replace('.csv', '.json').replace('.xlsx', '.json').replace('.xls', '.json').replace('.parquet', '.json').replace('.yaml', '.json').replace('.yml', '.json'),
                        mime="application/json",
                        use_container_width=True
                    )
                elif output_format == "YAML":
                    # Convert DataFrame to YAML format
                    # Replace NaN with None for YAML serialization
                    data_list = df_for_download.to_dict('records')
                    yaml_data = {
                        'metadata': {
                            'rows': len(df_for_download),
                            'columns': len(df_for_download.columns),
                            'column_names': list(df_for_download.columns)
                        },
                        'data': data_list
                    }
                    yaml_str = yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True, sort_keys=False)
                    st.download_button(
                        label="📥 Download YAML",
                        data=yaml_str,
                        file_name=st.session_state['processed_file_name'].replace('.csv', '.yaml').replace('.xlsx', '.yaml').replace('.xls', '.yaml').replace('.json', '.yaml').replace('.parquet', '.yaml'),
                        mime="application/x-yaml",
                        use_container_width=True
                    )
    
    # Results section
    if 'processed_df' in st.session_state:
        st.markdown("---")
        st.header("📊 Processing Results")
        
        processed_df = st.session_state['processed_df']
        
        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{processed_df.shape[0]:,}")
        
        with col2:
            st.metric("Columns", processed_df.shape[1])
        
        with col3:
            missing_after = processed_df.isnull().sum().sum()
            st.metric("Missing Values", missing_after)
        
        with col4:
            duplicates_after = processed_df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates_after)
        
        # Processed data preview
        st.subheader("Processed Data Preview")
        # Make DataFrame Arrow-compatible before display
        df_display = make_arrow_compatible(processed_df.head(20))
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Processing Summary Block
        if 'processing_summary' in st.session_state:
            summary = st.session_state['processing_summary']
            st.subheader("📊 Processing Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Missing Values Handled", f"{summary.get('missing_values_handled', 0):,}")
            with col2:
                st.metric("Duplicates Removed", f"{summary.get('duplicates_removed', 0):,}")
            with col3:
                st.metric("Numeric Columns Validated", summary.get('numeric_columns_validated', 0))
            with col4:
                st.metric("Text Columns Cleaned", summary.get('text_columns_cleaned', 0))
            
            # Summary box
            st.markdown("""
            <div class="success-box">
                <strong>Processing Summary:</strong><br>
                ✔ {missing:,} missing values handled<br>
                ✔ {duplicates:,} duplicates removed<br>
                ✔ {numeric} numeric columns validated<br>
                ✔ {text} text columns cleaned
            </div>
            """.format(
                missing=summary.get('missing_values_handled', 0),
                duplicates=summary.get('duplicates_removed', 0),
                numeric=summary.get('numeric_columns_validated', 0),
                text=summary.get('text_columns_cleaned', 0)
            ), unsafe_allow_html=True)
        
        # Validation results
        if 'validation_results' in st.session_state and st.session_state['validation_results']:
            st.subheader("✅ Validation Results")
            for result in st.session_state['validation_results']:
                status_emoji = "✅" if result['status'] == 'passed' else "❌"
                st.write(f"{status_emoji} **{result['check']}**: {result['status']}")
                
                # Show detailed results if available
                if 'detailed_results' in result and result['detailed_results']:
                    with st.expander(f"View detailed {result['check']} results"):
                        # Create a DataFrame for better display
                        df_details = pd.DataFrame(result['detailed_results'])
                        df_details_display = make_arrow_compatible(df_details)
                        st.dataframe(df_details_display, use_container_width=True)
                
                if 'errors' in result and result['errors']:
                    with st.expander(f"View {result['check']} errors"):
                        for error in result['errors'][:10]:  # Show first 10 errors
                            st.error(error)
        
        # Statistics
        with st.expander("📈 Processed Data Statistics"):
            df_stats = processed_df.describe()
            df_stats_display = make_arrow_compatible(df_stats)
            st.dataframe(df_stats_display, use_container_width=True)
        
        # Clear results button
        if st.button("🔄 Clear Results & Start Over", use_container_width=True, type="secondary"):
            for key in ['processed_df', 'processed_file_name', 'validation_results', 'processing_summary', 'df_preview', 'file_uploaded']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
