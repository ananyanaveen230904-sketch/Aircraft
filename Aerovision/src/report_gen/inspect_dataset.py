"""
Sanity-check the generated report_training_pairs.jsonl before spending
time fine-tuning on it.

Prints 10 random (input_text, target_text) examples side by side, plus
basic stats: total record count, avg/min/max target_text word length,
and the distribution of failure_risk phrases across generated inputs.

Run:
    python src/report_gen/inspect_dataset.py --data data/report_training_pairs.jsonl
"""

import argparse
import json
import random
from collections import Counter


def load_pairs(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/report_training_pairs.jsonl")
    parser.add_argument("--n", type=int, default=10, help="Number of random examples to print")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    if not pairs:
        print(f"No records found in {args.data}. Run generate_dataset.py first.")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(pairs, min(args.n, len(pairs)))

    print("=" * 90)
    print(f"{len(sample)} random example(s) out of {len(pairs)} total records")
    print("=" * 90)
    for row in sample:
        print(f"\n[record_index {row.get('record_index')}]")
        print(f"INPUT:  {row['input_text']}")
        print(f"TARGET: {row['target_text']}")
        print("-" * 90)

    # Basic stats
    target_word_counts = [len(row["target_text"].split()) for row in pairs]
    avg_len = sum(target_word_counts) / len(target_word_counts)

    risk_counter = Counter()
    defect_count_counter = Counter()
    no_rul_count = 0
    no_defects_count = 0

    for row in pairs:
        text = row["input_text"]
        if "Failure risk: High" in text:
            risk_counter["High"] += 1
        elif "Failure risk: Medium" in text:
            risk_counter["Medium"] += 1
        elif "Failure risk: Low" in text:
            risk_counter["Low"] += 1
        else:
            risk_counter["(none / partial record)"] += 1
            no_rul_count += 1

        if "Detected defects: none." in text:
            no_defects_count += 1
            defect_count_counter[0] += 1
        else:
            # rough defect count: count ", " + 1 in the defects clause
            defects_clause = text.split("Detected defects:")[-1]
            n_defects = defects_clause.count("% confidence)")
            defect_count_counter[n_defects] += 1

    print("\n" + "=" * 90)
    print("STATS")
    print("=" * 90)
    print(f"Total records:                {len(pairs)}")
    print(f"Avg target length (words):    {avg_len:.1f}")
    print(f"Min / Max target length:      {min(target_word_counts)} / {max(target_word_counts)}")
    print(f"\nRecords with no RUL/health data (partial): {no_rul_count} ({no_rul_count/len(pairs):.0%})")
    print(f"Records with zero defects:                 {no_defects_count} ({no_defects_count/len(pairs):.0%})")

    print(f"\nfailure_risk distribution:")
    for label, count in risk_counter.most_common():
        print(f"  {label:<28}{count:>5}  ({count/len(pairs):.0%})")

    print(f"\ndefect count distribution:")
    for n_defects, count in sorted(defect_count_counter.items()):
        print(f"  {n_defects} defect(s):{'':<15}{count:>5}  ({count/len(pairs):.0%})")


if __name__ == "__main__":
    main()
