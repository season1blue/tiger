import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=str)
    parser.add_argument("--dst", required=True, type=str)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    data = json.loads(src.read_text())

    # Input format: {"v1_0": {"imagename": ..., "question": ...}, ...}
    rows = []
    for key, item in data.items():
        if not key.startswith("v1_"):
            continue
        qid = int(key.split("_")[1])
        rows.append(
            {
                "question_id": qid,
                "image": item["imagename"],
                "text": item["question"],
            }
        )

    rows.sort(key=lambda x: x["question_id"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Converted {len(rows)} MM-Vet questions -> {dst}")


if __name__ == "__main__":
    main()
