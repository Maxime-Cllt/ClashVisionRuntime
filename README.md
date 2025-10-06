<div align="center">
    <h1>ClashVisionRuntime</h1>
    <p><em>High-performance CSV data validation and anomaly detection tool</em></p>
</div>
<div align="center">
  <!-- Rust -->
    <img src="https://img.shields.io/badge/Rust-dea584?style=for-the-badge&logo=rust&logoColor=white" alt="Rust" />

  <!-- ONNX Runtime -->
  <img src="https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime" />

  <!-- Version -->
  <img src="https://img.shields.io/badge/Version-1.0.0-6c63ff?style=for-the-badge" alt="Version" />

  <!-- License -->
  <img src="https://img.shields.io/badge/License-GPL--3.0-3C8DAD?style=for-the-badge&logo=open-source-initiative&logoColor=white" alt="License" />
</div>

## 🚀 Overview

**ClashVisionRuntime** is a production-ready

### ✨ Key Features

- 🔍 **AI-Powered Detection**: Load and run YOLOv8 model for object detection in images.
- ⚡ **High Performance**: Optimized for speed and efficiency using Rust.

## 📋 Prerequisites

### Required Tools

- **[Rust](https://www.rust-lang.org/tools/install)** (latest stable version)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Maxime-Cllt/ClashVisionRuntime.git
cd ClashVisionRuntime
```

### 2. Build the Project

```bash
# Development build
cargo build

# Optimized release build (recommended for production)
cargo build --release
```

## 🚀 Usage

### Command Line Interface

```bash
# Using cargo (development)
cargo run --release "input_file.csv" "output_report.json"

# Using compiled executable (production)
./target/release/DataLint "input_file.csv" "output_report.json"

# On Windows
.\target\release\DataLint.exe "input_file.csv" "output_report.json"
```

### Parameters

- **Input File**: Path to the CSV file to be validated
- **Output File**: Path where the JSON analysis report will be saved

### Example Usage

```bash
# Analyze a customer data file
./DataLint "data/customers.csv" "reports/customer_analysis.json"

# Validate uploaded user data
./DataLint "uploads/user_data.csv" "validation/results.json"
```

## 📊 Output Format

DataLint generates detailed JSON reports with the following structure:

## 🔧 Development

### Building from Source

To build ClashVisionRuntime from source, ensure you have Rust and Cargo installed, then run:

```bash
cargo build --release
```

## 🧪 Code quality

### Unit Tests available

The `tests` directory is tested using the command :

```bash
cargo test
```

### Benchmarking available

Code is benchmarked using the `criterion` crate. To run benchmarks, use:

```bash
cargo bench
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.