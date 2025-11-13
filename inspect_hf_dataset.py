#!/usr/bin/env python3
"""Inspect the COMI-LINGUA NER dataset from HuggingFace."""

from datasets import load_dataset

ds = load_dataset("LingoIITGN/COMI-LINGUA", "NER")

print("=== SPLITS ===")
for k, v in ds.items():
    print(f"{k}: len={len(v)}")

print("\n=== TRAIN COLUMNS ===")
print(ds["train"].column_names)

print("\n=== TRAIN FEATURES ===")
print(ds["train"].features)

print("\n=== FIRST 2 EXAMPLES ===")
for i in range(min(2, len(ds["train"]))):
    ex = ds["train"][i]
    # print keys and types, not full giant blobs
    print(f"\nExample {i}:")
    print({k: type(ex[k]).__name__ for k in ex})
    for k in ex:
        if isinstance(ex[k], list):
            print(f"  {k}: list len={len(ex[k])}, first3={ex[k][:3]}")
        else:
            s = str(ex[k])
            print(f"  {k}: {s[:120].replace(chr(10), ' ')}...")
