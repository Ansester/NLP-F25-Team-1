#!/usr/bin/env python3
"""Debug extraction logic."""

from datasets import load_dataset
import json

ds = load_dataset("LingoIITGN/COMI-LINGUA", "NER")
ex = ds["train"][0]

print("=== FULL FIRST EXAMPLE ===")
print(json.dumps({k: str(v)[:150] + "..." if isinstance(v, (str, list)) else v 
                  for k, v in ex.items()}, 
                 indent=2, ensure_ascii=False))

print("\n=== SENTENCES (tokens) ===")
print(f"Type: {type(ex['Sentences'])}")
print(f"Value: {ex['Sentences'][:150]}...")

print("\n=== ANNOTATED BY: ANNOTATOR 1 (tags) ===")
print(f"Type: {type(ex['Annotated by: Annotator 1'])}")
print(f"First 3 items: {ex['Annotated by: Annotator 1'][:3]}")
