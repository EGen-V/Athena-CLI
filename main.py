#!/usr/bin/env python3
import argparse
import importlib.util
import logging
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager

# =========================
# Logging configuration
# =========================
LOG_FILE = "athena_cli.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
fmt = logging.Formatter("%(message)s")
console.setFormatter(fmt)
logging.getLogger().addHandler(console)


# =========================
# Terminal styling (graceful if not a TTY)
# =========================
ANSI = {
    "bold": "\033[1m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "reset": "\033[0m",
}
if not sys.stdout.isatty():
    for k in ANSI:
        ANSI[k] = ""


# =========================
# Spinner / progress helpers
# =========================
class Spinner:
    """Simple ASCII spinner used while long-running tasks execute."""
    spinner_cycle = ["⠋", "⠙", "⠹", "⠸", "⢰", "⣄", "⣆", "⡇"]

    def __init__(self, label="Working", min_interval=0.08):
        self.label = label
        self._stop = threading.Event()
        self._thread = None
        self.min_interval = min_interval

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        idx = 0
        while not self._stop.is_set():
            char = self.spinner_cycle[idx % len(self.spinner_cycle)]
            sys.stdout.write(f"\r{ANSI['cyan']}{char} {self.label}{ANSI['reset']}")
            sys.stdout.flush()
            idx += 1
            time.sleep(self.min_interval)
        sys.stdout.write("\r" + " " * (len(self.label) + 8) + "\r")
        sys.stdout.flush()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)


@contextmanager
def show_progress(message="Processing..."):
    spinner = Spinner(message)
    spinner.start()
    try:
        yield
    finally:
        spinner.stop()


# =========================
# Auto-install helper (robust)
# =========================
def ensure_package(pkg_name, pip_name=None, retries=2, wait=3):
    """
    Ensure package is importable; if missing, try pip-install with limited retries.
    Returns True if package available, False otherwise.
    """
    pip_name = pip_name or pkg_name
    if importlib.util.find_spec(pkg_name) is not None:
        logging.debug("Package '%s' already present.", pkg_name)
        return True

    logging.info("[installer] Dependency '%s' not found. Attempting to install '%s'.", pkg_name, pip_name)
    for attempt in range(1, retries + 1):
        try:
            with show_progress(f"Installing {pip_name} (attempt {attempt}/{retries})"):
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # verify
            if importlib.util.find_spec(pkg_name) is not None:
                logging.info("[installer] Successfully installed '%s'.", pip_name)
                return True
            else:
                logging.warning("[installer] Installation finished but package '%s' still not importable.", pkg_name)
        except subprocess.CalledProcessError as e:
            logging.warning("[installer] pip failed on attempt %d for '%s'.", attempt, pip_name)
            logging.debug("pip error: %s", e)
        except Exception as e:
            logging.error("[installer] Unexpected error while installing '%s': %s", pip_name, e)
            logging.debug(traceback.format_exc())

        if attempt < retries:
            logging.info("[installer] Retrying in %d seconds...", wait)
            time.sleep(wait)

    logging.error("[installer] Could not install '%s' after %d attempts. Please install it manually and re-run.", pip_name, retries)
    return False


# =========================
# Model definitions and configurations
# =========================
AVAILABLE_MODELS = [
    {"id": "ErebusTN/EGen-SA1Q9", "name": "EGen-SA1Q9 (Recommended)", "description": "High-performance quantized model"},
    {"id": "ErebusTN/EGen-SA1Q8", "name": "EGen-SA1Q8 (Alternative)", "description": "Alternative quantized model"},
]

# =========================
# Required / optional packages
# =========================
REQUIRED = [
    ("torch", "torch"),
    ("transformers", "transformers>=4.54.0"),
    ("safetensors", "safetensors"),
    ("accelerate", "accelerate"),
    ("tqdm", "tqdm"),
]
OPTIONAL = [
    ("bitsandbytes", "bitsandbytes"),
    ("huggingface_hub", "huggingface_hub"),
]

def perform_auto_installs(allow_install=True):
    """Attempt to satisfy required packages. Returns (ok, missing_list)."""
    missing = []
    for pkg, pip_name in REQUIRED:
        if importlib.util.find_spec(pkg) is None:
            if allow_install:
                ok = ensure_package(pkg, pip_name)
                if not ok:
                    missing.append(pkg)
            else:
                missing.append(pkg)

    opt_missing = []
    for pkg, pip_name in OPTIONAL:
        if importlib.util.find_spec(pkg) is None:
            if allow_install:
                _ = ensure_package(pkg, pip_name) 
            else:
                opt_missing.append(pkg)

    return missing, opt_missing


# =========================
# Model/device helpers
# =========================
def choose_device_and_dtype(prefer_bf16_if_available=True):
    """Choose device/dtype; safe defaults if torch not available."""
    try:
        import torch 
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        dtype = None
        use_8bit = importlib.util.find_spec("bitsandbytes") is not None and use_cuda
        if use_cuda:
            if prefer_bf16_if_available and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                dtype = getattr(torch, "bfloat16", None)
            else:
                dtype = getattr(torch, "float16", None)
        else:
            dtype = getattr(torch, "float32", None)
        return device, dtype, use_8bit
    except Exception:
        logging.exception("Failed to inspect torch environment; defaulting to CPU.")
        return "cpu", None, False


def load_model_and_tokenizer(model_id, device, dtype, load_in_8bit=False):
    """
    Load tokenizer and model with friendly progress messages.
    Uses spinner while the transformers library retrieves model files.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        logging.error("transformers library not available. %s", e)
        raise

    logging.info("[loader] Preparing tokenizer for '%s'...", model_id)
    with show_progress("Loading tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model_kwargs = {"device_map": "auto"}
    if dtype is not None and device == "cuda":
        model_kwargs["torch_dtype"] = dtype
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    logging.info("[loader] Loading model (%s). This may take several minutes depending on your connection and cache state.", model_id)
    with show_progress("Downloading and initializing model"):
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    try:
        import torch
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    logging.info("[loader] Model loaded successfully.")
    return tokenizer, model


# =========================
# Generation helper
# =========================
def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, repetition_penalty=1.0):
    import time as _time
    start = _time.time()
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        from transformers import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        with __torch_inference_mode(model):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        out_ids = outputs[0]
        generated_ids = out_ids[input_ids.shape[-1]:] if out_ids.shape[-1] > input_ids.shape[-1] else []
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip() if len(generated_ids) else ""
        took = _time.time() - start
        return generated_text, took
    except Exception:
        logging.exception("Generation failed.")
        raise


@contextmanager
def __torch_inference_mode(model):
    """Context manager that uses torch.inference_mode() if available, else torch.no_grad()."""
    try:
        import torch
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            yield
    except Exception:
        yield


# =========================
# REPL / CLI
# =========================
ASCII_HEADER = r"""
       d8888 888    888                                       .d8888b.  888      8888888 
      d88888 888    888                                      d88P  Y88b 888        888   
     d88P888 888    888                                      888    888 888        888   
    d88P 888 888888 88888b.   .d88b.  88888b.   8888b.       888        888        888   
   d88P  888 888    888 "88b d8P  Y8b 888 "88b     "88b      888        888        888   
  d88P   888 888    888  888 88888888 888  888 .d888888      888    888 888        888   
 d8888888888 Y88b.  888  888 Y8b.     888  888 888  888      Y88b  d88P 888        888   
d88P     888  "Y888 888  888  "Y8888  888  888 "Y888888       "Y8888P"  88888888 8888888 v1.0
                            By ErebusTN — The Athena Project
"""
SEPARATOR = "-" * 72


def print_header():
    print(ANSI["cyan"] + ASCII_HEADER + ANSI["reset"])
    print(ANSI["bold"] + "Athena CLI — reliable, informative, and user-friendly." + ANSI["reset"])
    print(SEPARATOR)


def choose_model():
    """Display model selection menu and return chosen model ID."""
    print(ANSI["bold"] + "\nSelect a model:" + ANSI["reset"])
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {ANSI['green']}{model['name']}{ANSI['reset']}")
        print(f"     {model['description']}")
    
    while True:
        try:
            choice = input(ANSI["bold"] + "\nEnter choice (1-" + str(len(AVAILABLE_MODELS)) + ") [1]: " + ANSI["reset"])
            choice = choice.strip() or "1"
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_MODELS):
                selected = AVAILABLE_MODELS[idx]
                print(f"[info] Selected: {selected['name']}")
                print(SEPARATOR)
                return selected["id"]
            else:
                print(f"[error] Please enter a number between 1 and {len(AVAILABLE_MODELS)}")
        except ValueError:
            print("[error] Invalid input. Please enter a number.")


def repl(model_id=None, allow_auto_install=True, skip_model_selection=False):
    print_header()
    if model_id is None and not skip_model_selection:
        model_id = choose_model()
    if model_id is None:
        model_id = AVAILABLE_MODELS[0]["id"]
    logging.info("Starting Athena CLI (model=%s).", model_id)

    missing, opt_missing = perform_auto_installs(allow_install=allow_auto_install)
    if missing:
        logging.error("Missing required packages: %s. Aborting. To skip auto-install, run with --no-install and install manually.", ", ".join(missing))
        print(ANSI["red"] + "Critical dependencies missing. See log file for details: " + LOG_FILE + ANSI["reset"])
        sys.exit(1)

    device, dtype, use_8bit = choose_device_and_dtype()
    logging.info("Environment: device=%s dtype=%s 8bit_supported=%s", device, dtype, use_8bit)
    print(f"[info] Device: {device} | dtype: {dtype} | 8-bit: {use_8bit}")

    try:
        tokenizer, model = load_model_and_tokenizer(model_id, device, dtype, load_in_8bit=use_8bit)
    except Exception as e:
        logging.exception("Failed to load model/tokenizer.")
        print(ANSI["red"] + "[error] Failed to load model/tokenizer. See log file for details: " + LOG_FILE + ANSI["reset"])
        print(ANSI["yellow"] + "Tip: verify your model id and internet connection, or pre-download the model to the cache." + ANSI["reset"])
        sys.exit(1)

    print(ANSI["green"] + "[ready] Model loaded and ready. Type your prompt and press Enter." + ANSI["reset"])
    print(ANSI["cyan"] + "Tip: Type '/help' for commands or 'exit' to quit." + ANSI["reset"])
    print(SEPARATOR)

    gen_params = {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
    }

    last_generated = ""
    prompt_history = []
    current_model = model_id

    try:
        while True:
            try:
                prompt = input(ANSI["bold"] + ">> " + ANSI["reset"])
            except (EOFError, KeyboardInterrupt):
                print("\n[exit] Exiting. Goodbye.")
                break

            if prompt is None:
                continue
            prompt = prompt.strip()
            if prompt == "":
                continue

            # exit commands
            if prompt.lower() in ("exit", "quit", "/exit", "/quit", "q"):
                print("[exit] Goodbye.")
                break

            # slash-commands
            if prompt.startswith("/"):
                parts = prompt.split()
                cmd = parts[0].lower()
                if cmd in ("/help", "/h"):
                    print(ANSI["cyan"] + "Commands:\n"
                          "  /help or /h           Show this message\n"
                          "  /params               Show generation parameters\n"
                          "  /len <n>              Set max_new_tokens (1-2048)\n"
                          "  /temp <float>         Set temperature (0.0-2.0)\n"
                          "  /top_p <float>        Set top_p (0.0-1.0)\n"
                          "  /model                Show available models\n"
                          "  /save <filename>      Save last generated output to file\n"
                          "  /history [n]          Show last n prompts (default: 20)\n"
                          "  /clear                Clear history\n"
                          "  /exit or exit         Exit the CLI\n" + ANSI["reset"])
                    continue
                if cmd == "/params":
                    print(ANSI["cyan"] + "Generation Parameters:" + ANSI["reset"])
                    for k, v in gen_params.items():
                        print(f"  {k:20s}: {v}")
                    print(f"  {'Model':20s}: {current_model}")
                    continue
                if cmd == "/len" and len(parts) > 1:
                    try:
                        val = int(parts[1])
                        if 1 <= val <= 2048:
                            gen_params["max_new_tokens"] = val
                            print(f"[param] max_new_tokens set to {gen_params['max_new_tokens']}")
                        else:
                            print("[param] max_new_tokens must be between 1 and 2048.")
                    except ValueError:
                        print("[param] Invalid integer for /len.")
                    continue
                if cmd == "/temp" and len(parts) > 1:
                    try:
                        val = float(parts[1])
                        if 0.0 <= val <= 2.0:
                            gen_params["temperature"] = val
                            print(f"[param] temperature set to {gen_params['temperature']}")
                        else:
                            print("[param] temperature must be between 0.0 and 2.0.")
                    except ValueError:
                        print("[param] Invalid float for /temp.")
                    continue
                if cmd == "/top_p" and len(parts) > 1:
                    try:
                        val = float(parts[1])
                        if 0.0 <= val <= 1.0:
                            gen_params["top_p"] = val
                            print(f"[param] top_p set to {gen_params['top_p']}")
                        else:
                            print("[param] top_p must be between 0.0 and 1.0.")
                    except ValueError:
                        print("[param] Invalid float for /top_p.")
                    continue
                if cmd == "/save" and len(parts) > 1:
                    filename = parts[1]
                    try:
                        with open(filename, "w", encoding="utf-8") as fh:
                            fh.write(last_generated or "")
                        print(f"[save] Output written to {filename}")
                    except Exception as e:
                        logging.exception("Failed to save output to %s", filename)
                        print(ANSI["red"] + f"[error] Failed to save to {filename}. See log for details." + ANSI["reset"])
                    continue
                if cmd == "/history":
                    limit = 20
                    if len(parts) > 1:
                        try:
                            limit = int(parts[1])
                        except ValueError:
                            pass
                    shown = prompt_history[-limit:]
                    if not shown:
                        print("[history] No prompts in history.")
                    else:
                        print(ANSI["cyan"] + f"Last {len(shown)} prompts:" + ANSI["reset"])
                        for i, p in enumerate(shown, start=1):
                            print(f"  {i:3d}. {p[:70]}{'...' if len(p) > 70 else ''}")
                    continue
                if cmd == "/model":
                    print(ANSI["cyan"] + "Available Models:" + ANSI["reset"])
                    for i, model in enumerate(AVAILABLE_MODELS, 1):
                        status = "[ACTIVE]" if model["id"] == current_model else ""
                        print(f"  {i}. {model['name']} {status}")
                    print(ANSI["yellow"] + "Note: Switching models requires restarting the CLI." + ANSI["reset"])
                    continue
                if cmd == "/clear":
                    prompt_history.clear()
                    print("[history] History cleared.")
                    continue

                print("[help] Unknown command. Type /help for available commands.")
                continue

            # Normal prompt: generate
            prompt_history.append(prompt)
            try:
                generated, took = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=gen_params["max_new_tokens"],
                    temperature=gen_params["temperature"],
                    top_p=gen_params["top_p"],
                    repetition_penalty=gen_params["repetition_penalty"],
                )
            except Exception:
                logging.exception("Generation failed.")
                print(ANSI["red"] + "[error] Generation failed. See log for details." + ANSI["reset"])
                continue

            last_generated = generated
            if generated:
                print(ANSI["green"] + "\n" + SEPARATOR + "\n--- Response ---" + ANSI["reset"])
                print(generated)
                print(ANSI["green"] + "\n" + SEPARATOR + ANSI["reset"])
            else:
                print(ANSI["yellow"] + "[warning] Model produced no continuation." + ANSI["reset"])

            # approximate token count (best-effort)
            try:
                token_count = len(tokenizer(generated)["input_ids"]) if generated else 0
            except Exception:
                token_count = "N/A"

            tokens_per_sec = f"{token_count / took:.1f}" if took > 0 and token_count != "N/A" else "N/A"
            print(f"[info] Tokens: {token_count} | Speed: {tokens_per_sec} tok/s | Time: {took:.2f}s")
            print()

    finally:
        # cleanup
        try:
            import torch
            if torch.cuda.is_available():
                with show_progress("Freeing GPU memory"):
                    torch.cuda.empty_cache()
        except Exception:
            pass
        logging.info("Session closed cleanly.")
        print(ANSI["green"] + "\n[exit] Thank you for using Athena CLI!" + ANSI["reset"])


# =========================
# Entry point (argparse)
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Athena CLI — Interactive model interface with multiple model support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available models:\n" +
               "\n".join(f"  • {m['id']:30s} - {m['description']}" for m in AVAILABLE_MODELS)
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model identifier (Hugging Face repo id). If omitted, you'll be prompted to choose."
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not attempt to auto-install missing packages"
    )
    parser.add_argument(
        "--no-select",
        action="store_true",
        help="Skip model selection menu (use default or --model argument)"
    )
    args = parser.parse_args()

    try:
        repl(
            model_id=args.model,
            allow_auto_install=not args.no_install,
            skip_model_selection=args.no_select
        )
    except Exception:
        logging.exception("Fatal error in main.")
        print(ANSI["red"] + "A fatal error occurred. Please consult the log file: " + LOG_FILE + ANSI["reset"])
        sys.exit(2)


if __name__ == "__main__":
    main()
