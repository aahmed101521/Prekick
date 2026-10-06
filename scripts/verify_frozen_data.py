from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_HASHES = {
    Path("data/processed/model_data.csv"): (
        "53a49e6459909044004e68bae5c30d5c225f00a5458a67b17156f94783d29886"
    ),
    Path("data/processed/elo_history.csv"): (
        "b14a7b0a5bcc299739b981bbaf7136680603d780b2deda2e9dbafd18da3f5aac"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []

    for path, expected_hash in EXPECTED_HASHES.items():
        if not path.exists():
            failures.append(f"Missing frozen production artifact: {path}")
            continue

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            failures.append(
                "\n".join(
                    [
                        f"Frozen production artifact changed: {path}",
                        f"Expected SHA-256: {expected_hash}",
                        f"Actual SHA-256:   {actual_hash}",
                    ]
                )
            )
            continue

        print(f"Verified: {path}")
        print(f"SHA-256:  {actual_hash}")

    if failures:
        raise SystemExit(
            "\n\n".join(
                [
                    "PREKICK V1 FROZEN DATA INTEGRITY CHECK FAILED",
                    *failures,
                ]
            )
        )

    print()
    print("Prekick v1 frozen data integrity: PASS")


if __name__ == "__main__":
    main()
