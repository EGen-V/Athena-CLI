<div align="center">
  
<h1 style="font-size: 2em;">🏛️ Athena CLI 🐼</h1>
  
**Interactive command-line interface for advanced reasoning and language understanding with the Athena Project models.**

<img src="https://i.ibb.co/5X8LK2TQ/ATHENA-PROJECT.png" alt="Athena Project" width="500" height="600">

[Explore EGen-SA1Q8](https://huggingface.co/ErebusTN/EGen-SA1Q8) • [Explore EGen-SA1Q9](https://huggingface.co/ErebusTN/EGen-SA1Q9) • [Report Issue](https://huggingface.co/ErebusTN/EGen-SA1Q9/discussions) • [Creator Profile](https://huggingface.co/ErebusTN)<br>

The CLI is engineered with a cutting-edge software stack to ensure stability, performance, and seamless model interaction:

![Transformers](https://img.shields.io/badge/Transformers-4.54.0+-yellow?logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0+-red?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-green?logo=apache&logoColor=white)

</div>

---

## 📖 Overview

The **Athena CLI** is a sophisticated, user-friendly command-line interface designed to interact with the **Athena Project** models (**EGen-SA1Q9** and **EGen-SA1Q8**). Developed by **ErebusTN** in 2025, this tool provides a seamless gateway to advanced language model capabilities with intuitive command management and real-time performance monitoring.

The CLI abstracts away technical complexity while maintaining full control over generation parameters, making it ideal for researchers, developers, and power users who want to interact with state-of-the-art language models.

## 🌟 Key Features

* **🔄 Multi-Model Support:** Choose between EGen-SA1Q9 (Recommended) and EGen-SA1Q8 (Alternative)
* **⚡ Interactive CLI:** Real-time generation with customizable parameters
* **📊 Performance Metrics:** Token-per-second tracking and execution timing
* **💾 Session Management:** Built-in history tracking and output saving
* **🎯 Fine-Grained Control:** Adjust temperature, top-p, token limits, and more
* **🚀 Auto-Installation:** Automatic dependency resolution and installation
* **🖥️ Cross-Platform:** Works on Linux, macOS, and Windows
* **💯 Production-Ready:** Comprehensive logging and error handling

## 🛠️ System Requirements

### Minimum Requirements
- **Python:** 3.8 or higher
- **RAM:** 16GB (8GB with 8-bit quantization support)
- **Disk Space:** 10GB for model downloads and cache
- **OS:** Linux, macOS, or Windows

### Recommended Requirements
- **GPU:** NVIDIA GPU with CUDA support (e.g., RTX 3080+)
- **VRAM:** 8GB+ (for full-precision), 4GB+ (with quantization)
- **RAM:** 32GB
- **Python:** 3.10+

### Core Dependencies
```
torch>=2.0.0
transformers>=4.54.0
safetensors
accelerate
tqdm
```

### Optional Dependencies
```
bitsandbytes      # 8-bit quantization support
huggingface_hub   # Advanced model management
```

---

## 📥 Installation

### Step 1: Clone or Download
```bash
# Clone the repository or download the main.py file
git clone https://github.com/EGen-V/Athena-CLI.git
cd Athena-CLI
```

### Step 2: Set Up Python Environment (Recommended)
```bash
# Create a virtual environment
python -m venv athena_env

# Activate it
# On Linux/macOS:
source athena_env/bin/activate
# On Windows:
athena_env\Scripts\activate
```

### Step 3: Run the CLI
```bash
python main.py
```

The CLI will automatically:
- Detect your system configuration
- Install missing dependencies (if permitted)
- Prompt you to select a model
- Initialize and load the model

---

## 🚀 Quick Start

### Basic Usage
```bash
# Launch with interactive model selection
python main.py

# Launch with specific model
python main.py ErebusTN/EGen-SA1Q9

# Skip model selection menu
python main.py --no-select

# Skip auto-install of dependencies
python main.py --no-install
```

### First Interaction
1. The CLI displays the Athena header and system information
2. If multiple models are available, you'll be prompted to choose one
3. The model loads automatically (this may take 2-5 minutes)
4. Once ready, type your prompt and press Enter
5. The model generates a response with performance metrics

### Example Session
```
[info] Device: cuda | dtype: torch.float16 | 8-bit: True
[ready] Model loaded and ready. Type your prompt and press Enter.
Tip: Type '/help' for commands or 'exit' to quit.
------------------------------------------------------------------------

>> Explain quantum computing in simple terms.

------------------------------------------------------------------------
--- Response ---
Quantum computing harnesses the principles of quantum mechanics to process 
information in fundamentally different ways than classical computers...
------------------------------------------------------------------------

[info] Tokens: 87 | Speed: 45.3 tok/s | Time: 1.92s
```

---

## 📋 Command Reference

### Help & Information

| Command | Usage | Purpose |
|---------|-------|---------|
| `/help`, `/h` | `/help` | Display all available commands |
| `/params` | `/params` | Show all generation parameters and active model |
| `/model` | `/model` | List available models and current selection |

### Parameter Configuration

| Command | Usage | Range | Example |
|---------|-------|-------|---------|
| `/len` | `/len <n>` | 1-2048 | `/len 300` |
| `/temp` | `/temp <value>` | 0.0-2.0 | `/temp 0.9` |
| `/top_p` | `/top_p <value>` | 0.0-1.0 | `/top_p 0.95` |

### Session Management

| Command | Usage | Purpose |
|---------|-------|---------|
| `/save` | `/save <filename>` | Save the last generated response to a file |
| `/history` | `/history [n]` | Show last n prompts (default: 20) |
| `/clear` | `/clear` | Clear the prompt history |
| `exit`, `/exit` | `exit` | Gracefully exit the CLI |

### Generation Parameter Explanation

- **max_new_tokens (max_length):** Maximum number of tokens to generate (1-2048)
  - Lower = faster, more concise responses
  - Higher = longer, more detailed responses
  - Default: 200

- **temperature:** Controls randomness in generation (0.0-2.0)
  - 0.0 = Deterministic, always same response
  - 0.7 = Balanced (default)
  - 1.5+ = More creative and random
  
- **top_p (nucleus sampling):** Controls diversity via probability mass (0.0-1.0)
  - 0.9 = Consider words that make up 90% of probability mass (default)
  - Lower = More focused, conservative
  - Higher = More diverse, creative

---

## 💡 Usage Examples

### Example 1: Code Generation
```
>> /len 500
[param] max_new_tokens set to 500

>> Write a Python function to calculate Fibonacci numbers.

------------------------------------------------------------------------
--- Response ---
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
------------------------------------------------------------------------

[info] Tokens: 48 | Speed: 52.1 tok/s | Time: 0.92s
```

### Example 2: Creative Writing
```
>> /temp 1.2
[param] temperature set to 1.2

>> Write a short sci-fi story opening.

------------------------------------------------------------------------
--- Response ---
The neon rain fell sideways, defying gravity itself. Dr. Chen watched from
her tower as the antigrav fields stuttered above the megacity...
------------------------------------------------------------------------

[info] Tokens: 42 | Speed: 48.7 tok/s | Time: 0.86s
```

### Example 3: Analytical Response
```
>> /temp 0.3
[param] temperature set to 0.3

>> Analyze the pros and cons of renewable energy.

------------------------------------------------------------------------
--- Response ---
Renewable Energy Pros:
- Sustainable and infinite resources
- Lower operational costs over time
- Reduced environmental impact
...
------------------------------------------------------------------------

[info] Tokens: 127 | Speed: 51.2 tok/s | Time: 2.48s
```

### Example 4: History & Session Management
```
>> /history 5
[cyan]Last 5 prompts:
  1. Explain quantum computing in simple terms.
  2. Write a Python function to calculate Fibonacci numbers.
  3. Write a short sci-fi story opening.
  4. Analyze the pros and cons of renewable energy.
  5. What is machine learning?

>> /save response.txt
[save] Output written to response.txt

>> /clear
[history] History cleared.
```

---

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Set default model (optional)
export ATHENA_MODEL="ErebusTN/EGen-SA1Q9"

# Set cache directory for models
export HF_HOME="/path/to/cache"

# Run with custom logging
ATHENA_LOG_LEVEL=DEBUG python main.py
```

### Memory Optimization

For systems with limited VRAM, consider these strategies:

1. **Use Quantization:**
   ```bash
   python main.py  # Automatically uses 8-bit if available
   ```

2. **Reduce Token Limits:**
   ```
   >> /len 100  # Smaller generations
   ```

3. **Use Alternative Model:**
   ```bash
   python main.py ErebusTN/EGen-SA1Q8
   ```

### GPU Acceleration

The CLI automatically detects NVIDIA GPUs:
- Enables CUDA if available
- Uses float16 precision by default
- Enables 8-bit quantization if `bitsandbytes` is installed
- Sets `cudnn.benchmark = True` for performance

---

## 📊 Performance Monitoring

The CLI displays real-time performance metrics:

```
[info] Tokens: 87 | Speed: 45.3 tok/s | Time: 1.92s
       ↑           ↑                    ↑
   Generated    Speed in              Total
   token count  tokens/second         time
```

**Performance Benchmarks (RTX 3090):**
- EGen-SA1Q9: ~40-50 tok/s
- EGen-SA1Q8: ~45-55 tok/s

---

## 📝 Logging

The CLI maintains a comprehensive log file: `athena_cli.log`

### Log Levels
- **INFO:** Standard operational messages
- **WARNING:** Non-critical issues and warnings
- **ERROR:** Critical failures and stack traces
- **DEBUG:** Detailed diagnostic information

### View Recent Logs
```bash
# Last 20 lines
tail -20 athena_cli.log

# Search for errors
grep ERROR athena_cli.log

# Follow logs in real-time
tail -f athena_cli.log
```

---

## 🐛 Troubleshooting

### Issue: "Package 'torch' not found"
**Solution:** Let the CLI auto-install dependencies
```bash
python main.py  # Don't use --no-install
```

### Issue: "CUDA out of memory"
**Solution:** Use smaller token limits or enable quantization
```bash
# In CLI:
>> /len 100
# Or restart with alternative model:
python main.py ErebusTN/EGen-SA1Q8
```

### Issue: "Model downloads fail"
**Solution:** Pre-download model or check connectivity
```bash
# Pre-download model (run once):
python -c "from transformers import AutoModel; \
AutoModel.from_pretrained('ErebusTN/EGen-SA1Q9')"

# Check internet connection and Hugging Face access
```

### Issue: "Slow generation speed"
**Solution:** 
- Use GPU instead of CPU (`device: cuda`)
- Reduce `max_new_tokens`
- Close other applications consuming resources
- Clear GPU cache (automatic on exit, use `/clear` for history)

### Issue: "Import errors with specific packages"
**Solution:** Manually install required packages
```bash
pip install torch transformers safetensors accelerate tqdm
pip install bitsandbytes huggingface_hub  # Optional
```

---

## 📚 Model Information

### Available Models

| Model | Status | Description |
|-------|--------|-------------|
| **ErebusTN/EGen-SA1Q9** | ✅ Recommended | High-performance quantized model, optimal balance |
| **ErebusTN/EGen-SA1Q8** | ✅ Available | Alternative quantized variant |

### Training Details

- **Base Architecture:** Latest transformer optimizations (2025)
- **Training Methodology:** Supervised Fine-Tuning (SFT)
- **Training Framework:** TRL + PEFT
- **Quantization:** SA1Q9/Q8 designations indicate optimized weight distribution
- **Primary Datasets:**
  - EGen-Dataset (proprietary)
  - LMSYS Chat 1M
  - OpenLeecher cleaned dataset
  - CodeForces competitions
  - LeetCode problems
  - Magicoder OSS instructions

---

## 🔐 Privacy & Security

- **Local Processing:** All inference happens locally on your machine
- **No Data Transmission:** Prompts are never sent to external servers
- **Logging:** Only local logs are created (configurable)
- **Model Cache:** Downloaded to local Hugging Face cache directory

---

## 📜 License

This CLI tool is released under the **Apache 2.0 License**. The Athena Project models are also Apache 2.0 licensed.

---

## 🤝 Support & Contact

**Developed by ErebusTN**

- **Hugging Face Profile:** [@ErebusTN](https://huggingface.co/ErebusTN)
- **Model Repository:** [EGen-SA1Q9](https://huggingface.co/ErebusTN/EGen-SA1Q9)
- **Report Issues:** [Model Discussions](https://huggingface.co/ErebusTN/EGen-SA1Q9/discussions)
- **Project Year:** 2025

---

## 🎓 Citation

If you use the Athena CLI or models in your research, please cite:

```bibtex
@misc{athena_project_2025,
  title={Athena Project: Advanced Reasoning and Language Understanding},
  author={ErebusTN},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/ErebusTN/EGen-SA1Q9}}
}
```

---

## 📖 Additional Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library Guide](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs)
- [CUDA Setup Guide](https://developer.nvidia.com/cuda-toolkit)

---

<div align="center">
<i>Built with ❤️ By ErebusTN. Empowering advanced language understanding.</i>

**Version 1.0** • 2026
</div>
