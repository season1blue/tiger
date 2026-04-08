import argparse
import json
from pathlib import Path


def load_questions(path: Path):
    text = path.read_text().strip()
    if not text:
        return []

    # JSON array format
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data

    # JSONL format
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def dump_jsonl(items, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--limit", required=False, type=int, default=0)
    args = parser.parse_args()

    items = load_questions(Path(args.input))
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    dump_jsonl(items, Path(args.output))
    print(f"Prepared {len(items)} questions -> {args.output}")


if __name__ == "__main__":
    main()
