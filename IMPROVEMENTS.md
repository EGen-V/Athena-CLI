# Athena CLI Improvements

## Multiple Model Support
- Added `AVAILABLE_MODELS` list with two quantized models:
  - **ErebusTN/EGen-SA1Q9** (Recommended): High-performance quantized model
  - **ErebusTN/EGen-SA1Q8** (Alternative): Alternative quantized model

### Model Selection
- Interactive model selection menu on startup (can be skipped with `--no-select`)
- Users prompted to choose a model if not specified via CLI argument
- Model information displayed in `/params` command
- New `/model` command shows all available models and current active model

## Enhanced CLI Commands

### New Commands
- **`/top_p <float>`** - Set top_p sampling parameter (0.0-1.0)
- **`/model`** - Show available models and current selection
- **`/history [n]`** - Show last n prompts (default: 20, previously showed 50)
- **`/clear`** - Clear prompt history

### Improved Commands
- **`/len <n>`** - Now validates input (1-2048 range)
- **`/temp <float>`** - Now validates input (0.0-2.0 range)
- **`/top_p <float>`** - Now validates input (0.0-1.0 range)
- **`/params`** - Better formatted output with current model information
- **`/history [n]`** - Customizable limit, truncates long prompts to 70 chars
- **`/help`** - Updated with all new commands and parameter ranges

## Enhanced User Experience

### Better Output Formatting
- Response output now clearly separated with visual boundaries
- Green separators for improved readability
- Better visual hierarchy and structure

### Performance Metrics
- **Tokens per second (tok/s)** calculation and display
- Improved timing information in generation output
- Better error logging with exception details

### Improved Feedback
- Clearer status messages during model loading
- Better error handling with detailed logging
- Validation feedback for parameter changes
- Warning about model switching requiring restart

## Argument Parsing Improvements
- Added `--no-select` flag to skip model selection menu
- Model argument now optional (defaults to None, triggers menu)
- Better help text explaining model options
- Improved argument descriptions and epilog

## Performance Optimizations
- Better memory management in GPU cleanup
- More efficient logging
- Parameter validation before generation
- Cleaner session closing with proper cleanup

## Usage Examples

### Launch with Interactive Selection
```bash
python main.py
```

### Launch with Specific Model
```bash
python main.py ErebusTN/EGen-SA1Q9
```

### Skip Selection Menu
```bash
python main.py --no-select
```

### Skip Auto-Install
```bash
python main.py --no-install
```

### Combine Options
```bash
python main.py ErebusTN/EGen-SA1Q8 --no-install --no-select
```

## CLI Command Reference

| Command | Usage | Purpose |
|---------|-------|---------|
| `/help` | `/help` | Show available commands |
| `/params` | `/params` | Display all generation parameters and model |
| `/len` | `/len 300` | Set maximum output tokens (1-2048) |
| `/temp` | `/temp 0.8` | Set temperature (0.0-2.0) |
| `/top_p` | `/top_p 0.95` | Set top_p sampling (0.0-1.0) |
| `/model` | `/model` | Show available models |
| `/save` | `/save output.txt` | Save last response to file |
| `/history` | `/history 10` | Show last 10 prompts |
| `/clear` | `/clear` | Clear history |
| `exit` | `exit` | Quit the CLI |
