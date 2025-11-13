#!/usr/bin/env python3
"""Debug bootstrap extraction."""

from datasets import load_dataset
import json as json_module

ds = load_dataset("LingoIITGN/COMI-LINGUA", "NER")
ex = ds["train"][0]

print("=== Original Fields ===")
print("Keys:", list(ex.keys()))

print("\n=== Sentences (toks_raw) ===")
toks_raw = ex["Sentences"]
print(f"Type: {type(toks_raw)}, len={len(str(toks_raw))}")
print(f"Repr: {repr(toks_raw)[:150]}...")

print("\n=== Annotated by: Annotator 1 (labs_raw) ===")
labs_raw = ex["Annotated by: Annotator 1"]
print(f"Type: {type(labs_raw)}, len={len(str(labs_raw))}")
print(f"Repr: {repr(labs_raw)[:200]}...")

# Try parsing labs_raw
import ast as ast_module
try:
    parsed = ast_module.literal_eval(labs_raw)
    print(f"\nParsed labs_raw successfully, type: {type(parsed)}")
    if isinstance(parsed, list):
        print(f"List length: {len(parsed)}")
        if parsed:
            print(f"First item type: {type(parsed[0])}, value: {parsed[0]}")
except Exception as e:
    print(f"\nFailed to parse labs_raw: {e}")

# Token col detection
TOKEN_COL_PREF = ["tokens", "Tokens", "Sentences", "sentence", "Sentence"]
LABEL_COL_PREF = ["ner_tags", "labels", "BIO", "tags", "Annotated by: Annotator 1", "Predicted tags"]

tok_col = next((c for c in TOKEN_COL_PREF if c in ex), None)
lab_col = next((c for c in LABEL_COL_PREF  if c in ex), None)

print(f"\nDetected tok_col: {tok_col}")
print(f"Detected lab_col: {lab_col}")
