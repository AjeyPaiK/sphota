"""Sanskrit sandhi dataset building utilities."""

import os
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_TRANSLITERATION = True
except ImportError:
    print("indic-transliteration not found. Run: pip install sphota[transliteration]")
    HAS_TRANSLITERATION = False


def iast_to_devanagari(text):
    if not HAS_TRANSLITERATION or not text:
        return text
    try:
        return transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)
    except Exception:
        return text


def parse_misc(misc_str):
    result = {}
    if not misc_str or misc_str == "_":
        return result
    for part in misc_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def parse_conllu_file(filepath):
    sentences = []
    current_meta = {}
    current_tokens = []
    current_unsandhied = []
    in_sentence = False

    with open(filepath, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.startswith("## text:"):
                current_meta["text_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("## text_id:"):
                current_meta["text_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("## chapter:"):
                current_meta["chapter"] = line.split(":", 1)[1].strip()

            elif line.startswith("# text ="):
                current_meta["sandhi_text"] = line.split("=", 1)[1].strip()
                in_sentence = True
                current_tokens = []
                current_unsandhied = []

            elif line.startswith("# sent_id ="):
                current_meta["sent_id"] = line.split("=", 1)[1].strip()

            elif in_sentence and line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                token_id = parts[0]
                if "-" in token_id or "." in token_id:
                    continue
                form = parts[1]
                misc = parse_misc(parts[-1]) if len(parts) >= 10 else {}
                unsandhied = misc.get("Unsandhied", form)
                current_tokens.append(form)
                current_unsandhied.append(unsandhied)

            elif line == "" and in_sentence and current_tokens:
                if "sandhi_text" in current_meta:
                    sentences.append({
                        "text_name":   current_meta.get("text_name", ""),
                        "text_id":     current_meta.get("text_id", ""),
                        "chapter":     current_meta.get("chapter", ""),
                        "sent_id":     current_meta.get("sent_id", ""),
                        "sandhi_text": current_meta["sandhi_text"],
                        "tokens":      list(current_tokens),
                        "unsandhied":  list(current_unsandhied),
                        "vicchheda":   " ".join(current_unsandhied),
                    })
                in_sentence = False
                current_tokens = []
                current_unsandhied = []
                current_meta.pop("sandhi_text", None)
                current_meta.pop("sent_id", None)

    return sentences


def build_dataset(
    conllu_dir="sanskrit/dcs/data/conllu/files",
    out_dir="sandhi_dataset",
    max_files=None,
    min_tokens=2,
    max_tokens=50,
):
    """Build a sandhi dataset from CoNLL-U formatted files.

    Args:
        conllu_dir: Directory containing .conllu files
        out_dir: Output directory for generated datasets
        max_files: Maximum number of files to process (None for all)
        min_tokens: Minimum tokens per sentence
        max_tokens: Maximum tokens per sentence
    """
    import glob

    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(conllu_dir, "**", "*.conllu")
    all_files = sorted(glob.glob(pattern, recursive=True))
    if not all_files:
        print("No .conllu files found under: " + conllu_dir)
        return

    if max_files:
        all_files = all_files[:max_files]

    print("Found " + str(len(all_files)) + " .conllu files")

    all_pairs = []
    all_records = []
    skipped = 0

    for filepath in tqdm(all_files, desc="Parsing"):
        try:
            sentences = parse_conllu_file(filepath)
        except Exception as e:
            print("Error parsing " + filepath + ": " + str(e))
            continue

        for sent in sentences:
            n = len(sent["tokens"])
            if n < min_tokens or n > max_tokens:
                skipped += 1
                continue
            sandhi = sent["sandhi_text"].strip()
            vicchheda = sent["vicchheda"].strip()
            if not sandhi or not vicchheda:
                skipped += 1
                continue
            all_pairs.append((sandhi, vicchheda))
            all_records.append(sent)

    print("Extracted " + str(len(all_pairs)) + " pairs, skipped " + str(skipped))

    # Save IAST TSV
    iast_path = os.path.join(out_dir, "sandhi_pairs_iast.tsv")
    with open(iast_path, "w", encoding="utf-8") as f:
        f.write("sandhi\tvicchheda\n")
        for sandhi, vicchheda in all_pairs:
            f.write(sandhi + "\t" + vicchheda + "\n")
    print("IAST pairs saved: " + iast_path)

    # Save Devanagari TSV
    if HAS_TRANSLITERATION:
        dev_path = os.path.join(out_dir, "sandhi_pairs_devanagari.tsv")
        with open(dev_path, "w", encoding="utf-8") as f:
            f.write("sandhi\tvicchheda\n")
            for sandhi, vicchheda in tqdm(all_pairs, desc="Converting to Devanagari"):
                f.write(iast_to_devanagari(sandhi) + "\t" + iast_to_devanagari(vicchheda) + "\n")
        print("Devanagari pairs saved: " + dev_path)

    # Save JSON
    json_path = os.path.join(out_dir, "sandhi_pairs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print("JSON saved: " + json_path)

    # Save vocabulary
    if HAS_TRANSLITERATION:
        chars = set()
        for sandhi, vicchheda in all_pairs:
            chars.update(iast_to_devanagari(sandhi))
            chars.update(iast_to_devanagari(vicchheda))
        chars.discard(" ")
        vocab = ["<pad>", "<bos>", "<eos>", "<unk>", " "] + sorted(chars)
        vocab_path = os.path.join(out_dir, "vocab_devanagari.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for tok in vocab:
                f.write(tok + "\n")
        print("Vocab saved: " + vocab_path + " (" + str(len(vocab)) + " tokens)")

    # Print samples
    print("\nSAMPLE PAIRS (IAST)")
    for sandhi, vicchheda in all_pairs[:3]:
        print("  Sandhi:    " + sandhi)
        print("  Vicchheda: " + vicchheda)
        print()

    if HAS_TRANSLITERATION:
        print("SAMPLE PAIRS (Devanagari)")
        for sandhi, vicchheda in all_pairs[:3]:
            print("  Sandhi:    " + iast_to_devanagari(sandhi))
            print("  Vicchheda: " + iast_to_devanagari(vicchheda))
            print()

    # Stats
    lengths = [len(s[0].split()) for s in all_pairs]
    print("Total pairs:         " + str(len(all_pairs)))
    print("Avg sentence length: " + str(round(sum(lengths)/len(lengths), 1)))
    text_counts = Counter(r["text_name"] for r in all_records)
    print("\nTop 10 texts:")
    for text, count in text_counts.most_common(10):
        print("  " + str(count) + "  " + text)


def split_dataset(out_dir="sandhi_dataset", train_ratio=0.9, val_ratio=0.05, seed=42):
    """Split dataset into train/val/test splits.

    Args:
        out_dir: Directory containing the dataset files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        seed: Random seed for reproducibility
    """
    import random
    import glob

    random.seed(seed)

    for script in ["iast", "devanagari"]:
        tsv_path = os.path.join(out_dir, "sandhi_pairs_" + script + ".tsv")
        if not os.path.exists(tsv_path):
            continue
        with open(tsv_path, encoding="utf-8") as f:
            header = f.readline()
            lines = f.readlines()
        random.shuffle(lines)
        n = len(lines)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "train": lines[:n_train],
            "val":   lines[n_train:n_train + n_val],
            "test":  lines[n_train + n_val:],
        }
        for split_name, split_lines in splits.items():
            path = os.path.join(out_dir, split_name + "_" + script + ".tsv")
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                f.writelines(split_lines)
            print(split_name + " (" + script + "): " + str(len(split_lines)) + " pairs")
