#!/usr/bin/env python3
"""
Convert Audacity label file to LJSpeech metadata.csv.
Workflow:
  1) Record a long WAV in Audacity.
  2) Create a Label Track; add labels per utterance with the exact transcript.
  3) File -> Export -> Export Multiple...
     - Split based on labels, Name files by: "Numbering after file name prefix"
     - Set prefix to "utt"
     - WAV (PCM 16-bit), mono, 22050
  4) File -> Export -> Export Labels... -> labels.txt
  5) Run this script to generate metadata.csv from labels.txt
"""
import csv, sys, os

def convert(labels_path, out_csv="metadata.csv"):
    with open(labels_path, "r", encoding="utf-8") as f:
        rows = [l.strip().split("\t") for l in f if l.strip()]
    # Audacity label format: start_sec \t end_sec \t text
    # When exporting multiple, files are named utt0001.wav, utt0002.wav etc.
    meta = []
    idx = 1
    for r in rows:
        if len(r) < 3: 
            continue
        text = r[2].strip()
        utt_id = f"utt{idx:04d}"
        meta.append((utt_id, text))
        idx += 1
    with open(out_csv, "w", encoding="utf-8") as f:
        for utt, text in meta:
            f.write(f"{utt}|{text}\n")
    print(f"Wrote {out_csv} with {len(meta)} rows.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "labels.txt"
    convert(path)
