"""
Run this script ONCE from your project folder:
    python fix_tokenizer_config.py

It fixes the tokenizer_config.json in your fine-tuned Malayalam model directory
by converting extra_special_tokens from a list to a dict (which transformers expects).
"""

import json
import shutil
from pathlib import Path

MODEL_PATH = r"C:\projects\winner\model\whisper-ml-model\content\whisper-ml-finetuned-final"
CONFIG_FILE = Path(MODEL_PATH) / "tokenizer_config.json"

# ── Backup original ───────────────────────────────────────────────────────────
backup = CONFIG_FILE.with_suffix(".json.bak")
shutil.copy(CONFIG_FILE, backup)
print(f"✅ Backup saved to: {backup}")

# ── Load ──────────────────────────────────────────────────────────────────────
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# ── Fix extra_special_tokens ──────────────────────────────────────────────────
val = cfg.get("extra_special_tokens")
print(f"   extra_special_tokens type : {type(val).__name__}")
print(f"   extra_special_tokens value: {val}")

if isinstance(val, list):
    # transformers expects a dict {token_name: token_string}
    # For Whisper the list entries are just token strings with no names,
    # so we set it to an empty dict — the tokenizer already has these
    # tokens registered in its vocab/tokenizer.json and doesn't need
    # extra_special_tokens to re-declare them.
    cfg["extra_special_tokens"] = {}
    print("   ✅ Converted list → empty dict {}")
elif isinstance(val, dict):
    print("   ✅ Already a dict — no change needed")
else:
    cfg["extra_special_tokens"] = {}
    print(f"   ⚠️  Unexpected type {type(val).__name__} — reset to {{}}")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(CONFIG_FILE, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print(f"\n✅ Fixed tokenizer_config.json saved to:\n   {CONFIG_FILE}")
print("\nYou can now restart your Streamlit app and Malayalam transcription will work.")