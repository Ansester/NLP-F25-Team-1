#!/usr/bin/env python3
"""Debug the extraction for COMI-LINGUA."""

from datasets import load_dataset
import json as json_module

ds = load_dataset("LingoIITGN/COMI-LINGUA", "NER")
ex = ds["train"][0]

print("=== Tokens (Sentences) ===")
toks_raw = ex["Sentences"]
print(f"Type: {type(toks_raw)}, is str: {isinstance(toks_raw, str)}")
if isinstance(toks_raw, str):
    try:
        parsed = json_module.loads(toks_raw)
        print(f"Parsed successfully, type: {type(parsed)}")
        if isinstance(parsed, list):
            print(f"Is list, first item type: {type(parsed[0]) if parsed else 'empty'}")
    except Exception as e:
        print(f"Failed to parse as JSON: {e}")
        print(f"First 200 chars: {toks_raw[:200]}")

print("\n=== Tags (Annotated by: Annotator 1) ===")
labs_raw = ex["Annotated by: Annotator 1"]
print(f"Type: {type(labs_raw)}, is str: {isinstance(labs_raw, str)}")
if isinstance(labs_raw, str):
    try:
        parsed = json_module.loads(labs_raw)
        print(f"Parsed successfully, type: {type(parsed)}")
        if isinstance(parsed, list):
            print(f"Is list, first 3 items:")
            for i, item in enumerate(parsed[:3]):
                print(f"  [{i}] type={type(item)}, value={item}")
    except Exception as e:
        print(f"Failed to parse as JSON: {e}")
        print(f"First 200 chars: {labs_raw[:200]}")
