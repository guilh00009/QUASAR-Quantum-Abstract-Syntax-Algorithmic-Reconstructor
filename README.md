<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<img width="1024" height="1024" alt="Gemini_Generated_Image_75qrz575qrz575qr" src="https://github.com/user-attachments/assets/41174b76-2808-44f0-a5cd-851d97d7a415" />

<br />
<div align="center">
<h3 align="center">Comprehensive Python-to-Quantum Converter</h3>

  <p align="center">
    Transform Python scripts into quantum circuits using IBM Quantum and Qiskit
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This is a comprehensive converter that transforms **EVERY Python operation** into quantum circuits using IBM Quantum and Qiskit. The tool analyzes Python scripts through AST parsing and converts all operations including:

- **Arithmetic Operations**: Addition, subtraction, multiplication, division
- **Control Flow**: If/else statements, for/while loops
- **Functions**: Both built-in and user-defined functions
- **Variables**: Variable assignments and references
- **Print Statements**: Console output simulation
- **Complex Expressions**: Binary operations and comparisons

The converter generates optimized quantum circuits that can be executed on:
- Local Qiskit simulators
- IBM Quantum hardware and cloud simulators
- Any Qiskit-compatible backend

### Key Features

- **Complete Python-to-Quantum Translation**: Converts ALL Python operations to quantum gates
- **IBM Quantum Integration**: Native support for IBM Quantum Runtime and hardware
- **OpenQASM Export**: Generates IBM Quantum-compatible circuit files
- **Advanced Circuit Optimization**: Includes error correction and circuit optimization
- **Comprehensive Analysis**: Detailed quantum circuit metrics and performance analysis
- **Deterministic Results**: Ensures correct numerical results for arithmetic operations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [![Python][Python.js]][Python-url]
- [![Qiskit][Qiskit.js]][Qiskit-url]
- [![IBM Quantum][IBMQuantum.js]][IBMQuantum-url]
- [![NumPy][NumPy.js]][NumPy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Get started with the Comprehensive Python-to-Quantum Converter in just a few steps.

### Prerequisites

- Python 3.8 or higher
- IBM Quantum account (optional, for hardware execution)
- Basic understanding of quantum computing concepts

### Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/github_username/repo_name.git
   cd repo_name
   ```

2. **Install required packages**
   ```sh
   pip install qiskit qiskit-ibm-runtime numpy
   ```

3. **Set up IBM Quantum (optional)**
   ```sh
   # Set your IBM Quantum token as environment variable
   export IBM_QUANTUM_TOKEN="your_token_here"
   ```

4. **Verify installation**
   ```sh
   python comprehensive_quantum_converter.py demo
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

### Command Line Interface

The Comprehensive Python-to-Quantum Converter supports a powerful command-line interface with extensive configuration options:

#### Basic Command Structure
```sh
python comprehensive_quantum_converter.py [file] [options]
```

#### Core Options

| Option | Description | Default | Examples |
|--------|-------------|---------|----------|
| `--backend` | Execution backend (`local` or `ibm`) | `local` | `--backend ibm` |
| `--shots` | Number of quantum measurements | `1024` | `--shots 100`, `--shots 8192` |
| `--help` | Show help message | - | `--help` |

### Quick Start Examples

#### 1. Basic Local Simulation
```sh
# Convert main.py with default settings (local backend, 1024 shots)
python comprehensive_quantum_converter.py main.py

# Convert with custom shot count (safe for most systems)
python comprehensive_quantum_converter.py main.py --shots 512

# Convert with custom backend and shots
python comprehensive_quantum_converter.py main.py --backend ibm --shots 100
```

#### 2. IBM Quantum Hardware Execution
```sh
# Run on IBM Quantum hardware (requires IBM_QUANTUM_TOKEN)
python comprehensive_quantum_converter.py main.py --backend ibm --shots 2048

# Use IBM Quantum cloud simulator
python comprehensive_quantum_converter.py main.py --backend ibm --shots 4096
```

#### 3. Built-in Demo Mode
```sh
# Run the comprehensive demo (no file needed)
python comprehensive_quantum_converter.py demo

# Demo with custom shots (conservative for memory)
python comprehensive_quantum_converter.py demo --shots 256
```

### Python API Usage

#### Basic File Conversion
```python
from comprehensive_quantum_converter import convert_python_file

# Convert a Python file
result = convert_python_file("main.py", backend_mode="local", shots=1024)

print("Python Output:", result.python_output)
print("Quantum Output:", result.quantum_output)
print("Circuit Qubits:", result.circuit_info['num_qubits'])
```

#### Advanced Programmatic Usage
```python
from comprehensive_quantum_converter import ComprehensivePythonToQuantumConverter

# Create converter with IBM backend
converter = ComprehensivePythonToQuantumConverter(backend_mode="ibm")

# Convert Python code string
python_code = """
x = 5
y = 10
result = x + y
print(f"Sum: {result}")
"""

result = converter.convert_script(python_code, shots=1024)

# Access detailed results
print("Execution Time:", result.execution_time)
print("Quantum Algorithm:", result.quantum_algorithm)
print("Circuit Depth:", result.circuit_info['depth'])
print("QASM File:", result.circuit_info.get('qasm_file'))

# Variable analysis
for var_name, var_value in result.variable_values.items():
    print(f"Variable {var_name} = {var_value}")

# Quantum variable analysis
for var_name, var_value in result.quantum_variables.items():
    print(f"Quantum {var_name} = {var_value}")
```

### Backend Configuration Guide

#### Local Backend
- **Best for**: Development, testing, debugging
- **Speed**: Fast (seconds)
- **Cost**: Free
- **Accuracy**: Perfect simulation of quantum behavior
- **Use case**: Prototyping and learning

```sh
# Local simulation with 100 shots
python comprehensive_quantum_converter.py main.py --backend local --shots 100
```

#### IBM Quantum Backend
- **Best for**: Production, real quantum hardware
- **Speed**: Slower (minutes to hours)
- **Cost**: Free tier available, premium options
- **Accuracy**: Real quantum noise and decoherence
- **Use case**: Production quantum computing

```sh
# IBM Quantum execution (requires token)
export IBM_QUANTUM_TOKEN="your_token_here"
python comprehensive_quantum_converter.py main.py --backend ibm --shots 2048
```

### Shots Configuration Guide

#### Understanding Shots
- **Shots** = Number of times the quantum circuit is measured
- **Higher shots** = More accurate results, but slower execution
- **Lower shots** = Faster results, but less accurate statistics

#### Recommended Shot Counts
| Use Case | Recommended Shots | Reason |
|----------|------------------|---------|
| Quick testing | 100-500 | Fast feedback during development |
| Accurate results | 1024-4096 | Good balance of speed vs accuracy |
| Production | 8192+ | Maximum accuracy for critical applications |
| Statistical analysis | 16384+ | Detailed probability distributions |

#### Examples
```sh
# Quick development iteration
python comprehensive_quantum_converter.py main.py --shots 100

# Balanced production use
python comprehensive_quantum_converter.py main.py --backend ibm --shots 2048

# Maximum accuracy for research
python comprehensive_quantum_converter.py main.py --backend ibm --shots 8192
```

### Output Analysis

#### Understanding Results
After conversion, you'll receive:
1. **Python Output**: Original Python script execution results
2. **Quantum Output**: Quantum simulation results
3. **Circuit Information**: Quantum circuit metrics
4. **OpenQASM File**: Generated quantum circuit file (`.qasm`)

#### Sample Output Analysis
```
🐍 Python Output:
hello guys: 2

🐍 Quantum Output:
🎯 CORRECT RESULT: 2 (Python: 2, confidence: 85) | Result 2: 2 (confidence: 15)

📊 Circuit Information:
   Qubits: 8
   Depth: 12
   Operations: {'h': 8, 'cx': 6, 'measure': 8}
   OpenQASM File: quantum_code.qasm
```

### Advanced Examples

#### Converting Complex Python Code
```python
# Example: Complex arithmetic and control flow
code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(5)
print(f"Fibonacci(5) = {result}")
"""

result = converter.convert_script(code, shots=2048)
print("Quantum Fibonacci Result:", result.quantum_output)
```

#### Batch Processing Multiple Files
```sh
# Process multiple Python files
for file in *.py; do
    echo "Processing $file..."
    python comprehensive_quantum_converter.py "$file" --backend local --shots 512
done
```

#### Configuration with Environment Variables
```sh
# Set IBM Quantum token
export IBM_QUANTUM_TOKEN="your_ibm_token"

# Set default backend
export QUANTUM_BACKEND="ibm"

# Run with environment settings
python comprehensive_quantum_converter.py main.py --shots 1024
```

### Troubleshooting

#### Common Issues and Solutions

**1. IBM Quantum Token Issues**
```sh
# Error: "IBM Runtime not configured"
export IBM_QUANTUM_TOKEN="your_token_from_ibm_quantum"
python comprehensive_quantum_converter.py main.py --backend ibm
```

**2. Import Errors**
```sh
# Install missing dependencies
pip install qiskit qiskit-ibm-runtime numpy
```

**3. Memory Issues with Large Circuits**
```sh
# Reduce circuit complexity or increase shots
python comprehensive_quantum_converter.py large_script.py --shots 100

# For complex algorithms, use fewer shots initially
python comprehensive_quantum_converter.py complex_algorithm.py --shots 50
```

**4. Memory Allocation Errors**
```sh
# Error: "Unable to allocate X GiB for an array"
# Solution: Reduce the number of shots for complex circuits
python comprehensive_quantum_converter.py complex_script.py --shots 50

# Or use local backend with minimal shots for development
python comprehensive_quantum_converter.py complex_script.py --backend local --shots 25
```

**5. Slow Execution**
```sh
# Use local backend for faster results
python comprehensive_quantum_converter.py main.py --backend local --shots 512
```

#### Performance Optimization Tips

1. **For Development**: Use `--backend local --shots 50-100` (minimal memory usage)
2. **For Testing**: Use `--backend local --shots 512-1024` (balanced performance)
3. **For Production**: Use `--backend ibm --shots 2048+` (real quantum hardware)
4. **For Research**: Use `--backend ibm --shots 8192+` (maximum accuracy)
5. **For Complex Algorithms**: Start with `--shots 25` and gradually increase
6. **Memory-Constrained Systems**: Use `--backend local --shots 25-50`

### Generated Files

After successful conversion, the tool generates:
- `quantum_code.qasm`: OpenQASM circuit file
- `quantum_code.qpy`: QPY format file (advanced users)
- Console output with detailed analysis

These files can be used with IBM Quantum Composer or other quantum tools.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [x] Basic Python-to-Quantum conversion
- [x] IBM Quantum Runtime integration
- [x] OpenQASM file generation
- [x] Advanced circuit optimization
- [x] Error correction implementation
- [ ] Support for quantum machine learning operations
- [ ] Real-time quantum circuit visualization
- [ ] Support for custom quantum gates
- [ ] Performance benchmarking suite
- [ ] Web-based interface
- [ ] Integration with other quantum cloud platforms

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* [Qiskit](https://qiskit.org/) - The open-source quantum computing framework
* [IBM Quantum](https://quantum.cloud.ibm.com/) - Quantum computing cloud platform
* [Qiskit Runtime](https://qiskit.org/ecosystem/ibm-runtime/) - Quantum runtime service
* [NumPy](https://numpy.org/) - Fundamental package for array computing
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template) - README template

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->



[Python.js]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/
[Qiskit.js]: https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=qiskit&logoColor=white
[Qiskit-url]: https://qiskit.org/
[IBMQuantum.js]: https://img.shields.io/badge/IBM%20Quantum-052FAD?style=for-the-badge&logo=ibm&logoColor=white
[IBMQuantum-url]: https://quantum.cloud.ibm.com/
[NumPy.js]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
