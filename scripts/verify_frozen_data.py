from __future__ import annotations

import hashlib
from pathlib import Path


# These hashes use a canonical LF (\n) representation so that the integrity
# check is identical on Windows and Linux. Git may check out text files with
# CRLF on Windows and LF on Linux even when their actual data are unchanged.
EXPECTED_HASHES = {
    Path("data/processed/model_data.csv"): (
        "2a5c0280bd18992b9f906edec29f25467b0227a73ec90ba82c90fe0558127e24"
    ),
    Path("data/processed/elo_history.csv"): (
        "be0f360f49c841cca92b9e7a5892b0651951450a64daba362e76c02bb653953e"
    ),
}


def sha256_file(path: Path) -> str:
    data = path.read_bytes()

    # Canonicalize Windows CRLF line endings to LF before hashing.
    canonical_data = data.replace(b"\r\n", b"\n")

    return hashlib.sha256(canonical_data).hexdigest()


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
                        f"Expected canonical SHA-256: {expected_hash}",
                        f"Actual canonical SHA-256:   {actual_hash}",
                    ]
                )
            )
            continue

        print(f"Verified: {path}")
        print(f"Canonical SHA-256: {actual_hash}")

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
