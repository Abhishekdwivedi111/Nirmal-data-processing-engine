# Architecture & Design Patterns - Interview Guide

## 🏗️ Architecture Overview

If an interviewer asks about the architecture and file organization, here's how to explain it:

---

## 📐 Primary Architecture: **Modular/Layered Architecture with Separation of Concerns**

### **Core Principle**
I followed a **modular architecture** with clear **separation of concerns**, where each module has a single, well-defined responsibility.

### **Three-Layer Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
│  - app.py (Streamlit Web Interface)                      │
│  - main.py (CLI Interface)                               │
│  - Handles user interaction and I/O                      │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                  │
│  - engine.py (Orchestrator/Controller)                  │
│  - Coordinates all processing operations                 │
│  - Manages configuration and pipeline execution         │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ SERVICE LAYER│ │ SERVICE LAYER│ │ SERVICE LAYER│
│              │ │              │ │              │
│ cleaners.py  │ │ validators.py│ │preprocessors │
│              │ │              │ │    .py       │
│ Data Cleaning│ │  Validation  │ │ Preprocessing│
│  Operations  │ │  Operations  │ │  Operations  │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                  │
│  - config_loader.py (Configuration Management)          │
│  - logger_setup.py (Logging Infrastructure)             │
│  - File I/O operations                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 File Organization Strategy

### **1. Package-Based Organization (nirmal/)**

I organized the core functionality into a **Python package** (`nirmal/`) following Python best practices:

```
nirmal/
├── __init__.py          # Package initialization
├── engine.py            # Orchestrator (Controller)
├── cleaners.py          # Data Cleaning Service
├── validators.py        # Validation Service
├── preprocessors.py     # Preprocessing Service
├── config_loader.py     # Configuration Infrastructure
└── logger_setup.py      # Logging Infrastructure
```

**Why this organization?**
- **Modularity**: Each file has a single responsibility
- **Reusability**: Components can be imported and used independently
- **Testability**: Each module can be tested in isolation
- **Maintainability**: Easy to locate and modify specific functionality

### **2. Configuration-Driven Architecture**

```
config/
├── default_config.yaml    # Generic configuration
├── example_config.yaml    # Template with all options
├── test_config.yaml       # Test-specific configuration
└── airbnb_config.yaml     # Domain-specific example
```

**Design Pattern**: **Strategy Pattern** - Different configurations represent different strategies for data processing.

### **3. Separation of Entry Points**

```
app.py          # Web interface entry point (Streamlit)
main.py         # CLI entry point
test_nirmal.py  # Test entry point
```

**Why separate entry points?**
- **Single Responsibility**: Each entry point serves a specific use case
- **Flexibility**: Can add new interfaces (REST API, etc.) without modifying core logic
- **Clean Architecture**: Presentation layer separated from business logic

---

## 🎯 Design Patterns Used

### **1. Strategy Pattern**
**Where**: Configuration-based processing strategies
**Example**: Different outlier handling methods (IQR vs Z-score), different missing value strategies (fill vs drop)

```python
# In cleaners.py
def remove_outliers(self, df, columns, method='iqr', ...):
    if method == 'iqr':
        # IQR strategy
    elif method == 'zscore':
        # Z-score strategy
```

**Why**: Allows runtime selection of algorithms without changing code structure.

### **2. Template Method Pattern**
**Where**: Pipeline execution flow in `engine.py`
**Example**: The `process()` method defines the algorithm skeleton:

```python
def process(self, input_path, output_path):
    # Template: Always follows this structure
    df = self._load_data(input_path)      # Step 1
    df = self.cleaner.clean(df, ...)      # Step 2
    self.validator.validate(df, ...)      # Step 3
    df = self.preprocessor.preprocess(...) # Step 4
    self._save_data(df, output_path)      # Step 5
```

**Why**: Ensures consistent pipeline execution while allowing flexibility in each step.

### **3. Factory Pattern**
**Where**: Component initialization based on configuration
**Example**: Creating cleaners, validators, preprocessors based on config:

```python
# In engine.py __init__
self.cleaner = DataCleaner(self.config.get('cleaning', {}))
self.validator = DataValidator(self.config.get('validation', {}))
self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
```

**Why**: Decouples object creation from usage, making the system more flexible.

### **4. Dependency Injection**
**Where**: Configuration passed to components
**Example**: Each component receives its configuration as a parameter:

```python
class DataCleaner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
```

**Why**: Makes components testable and loosely coupled.

### **5. Facade Pattern**
**Where**: `DataProcessingEngine` acts as a facade
**Example**: Simple interface hiding complexity:

```python
engine = DataProcessingEngine(config_path='config.yaml')
df = engine.process('input.csv', 'output.csv')
# User doesn't need to know about cleaners, validators, etc.
```

**Why**: Provides a simple interface to a complex subsystem.

---

## 🏛️ Architectural Principles Followed

### **1. Separation of Concerns (SoC)**
- **Cleaners**: Only handle data cleaning
- **Validators**: Only handle validation
- **Preprocessors**: Only handle preprocessing
- **Engine**: Only orchestrates the pipeline

### **2. Single Responsibility Principle (SRP)**
Each class/module has one reason to change:
- `DataCleaner` changes only if cleaning logic changes
- `DataValidator` changes only if validation logic changes
- `DataPreprocessor` changes only if preprocessing logic changes

### **3. Open/Closed Principle (OCP)**
- **Open for extension**: Can add new cleaning methods, validators, preprocessors
- **Closed for modification**: Core pipeline structure doesn't change

### **4. Dependency Inversion Principle (DIP)**
- High-level modules (engine) don't depend on low-level modules (cleaners)
- Both depend on abstractions (configuration interfaces)

### **5. Don't Repeat Yourself (DRY)**
- Common functionality (file I/O, logging) centralized
- Configuration parsing abstracted into `config_loader.py`

---

## 📊 How to Explain in Interview

### **Short Answer (30 seconds)**

> "I followed a **modular, layered architecture** with clear separation of concerns. The project is organized into three layers: a presentation layer (web UI and CLI), a business logic layer (orchestrator), and a service layer (cleaners, validators, preprocessors). Each module has a single responsibility, making it easy to test, maintain, and extend. I used design patterns like Strategy Pattern for configurable algorithms, Template Method for the pipeline flow, and Facade Pattern to provide a simple interface to the complex system."

### **Detailed Answer (2-3 minutes)**

> "I designed the architecture following **modular and layered architecture principles** with **separation of concerns** as the core principle.
>
> **File Organization:**
> I organized the code into a Python package structure where:
> - The `nirmal/` package contains all core business logic
> - Each module (`cleaners.py`, `validators.py`, `preprocessors.py`) has a single, well-defined responsibility
> - The `engine.py` acts as an orchestrator that coordinates all operations
> - Configuration and infrastructure concerns are separated into dedicated modules
>
> **Three-Layer Architecture:**
> 1. **Presentation Layer**: `app.py` (Streamlit) and `main.py` (CLI) handle user interaction
> 2. **Business Logic Layer**: `engine.py` orchestrates the pipeline
> 3. **Service Layer**: Specialized modules for cleaning, validation, and preprocessing
>
> **Design Patterns:**
> I implemented several design patterns:
> - **Strategy Pattern**: For configurable algorithms (IQR vs Z-score, fill vs drop)
> - **Template Method Pattern**: For the consistent pipeline execution flow
> - **Factory Pattern**: For component initialization based on configuration
> - **Facade Pattern**: The engine provides a simple interface to a complex subsystem
> - **Dependency Injection**: Components receive configuration, making them testable
>
> **Benefits:**
> This architecture provides:
> - **Modularity**: Each component can be developed and tested independently
> - **Extensibility**: Easy to add new cleaning methods or validators
> - **Maintainability**: Clear separation makes it easy to locate and fix issues
> - **Testability**: Each module can be unit tested in isolation
> - **Flexibility**: Configuration-driven approach allows runtime behavior changes"

### **If Asked About Specific Design Decisions**

**Q: "Why did you separate cleaners, validators, and preprocessors?"**

> "I separated them following the **Single Responsibility Principle**. Each module has one reason to change:
> - Cleaners change only if cleaning logic changes
> - Validators change only if validation rules change
> - Preprocessors change only if preprocessing needs change
> This makes the codebase more maintainable and allows different team members to work on different modules without conflicts."

**Q: "Why use YAML configuration instead of hardcoding?"**

> "I used YAML configuration to implement the **Strategy Pattern** and **Open/Closed Principle**. This allows:
> - Runtime behavior changes without code modifications
> - Different configurations for different datasets
> - Version control of processing strategies
> - Easy experimentation with different cleaning approaches
> - Reproducible pipelines"

**Q: "How did you ensure the architecture is scalable?"**

> "I designed it with scalability in mind:
> - **Modular design**: Easy to add new processing modules
> - **Configuration-driven**: New strategies can be added via config
> - **Loose coupling**: Components communicate through well-defined interfaces
> - **Separation of concerns**: Changes in one area don't affect others
> - **Infrastructure abstraction**: Logging and configuration are abstracted, making it easy to swap implementations"

---

## 🎯 Key Takeaways for Interview

1. **Architecture Type**: Modular/Layered Architecture with Separation of Concerns
2. **Organization**: Package-based with single responsibility per module
3. **Patterns**: Strategy, Template Method, Factory, Facade, Dependency Injection
4. **Principles**: SOLID principles, especially SRP and SoC
5. **Benefits**: Modularity, Extensibility, Maintainability, Testability

---

## 📝 Quick Reference

**Architecture**: Modular/Layered Architecture  
**Patterns**: Strategy, Template Method, Factory, Facade, Dependency Injection  
**Principles**: SOLID (especially SRP, SoC, OCP)  
**Organization**: Package-based with clear separation of concerns  
**Key Design**: Configuration-driven, loosely coupled, highly cohesive modules
