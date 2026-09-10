"""Download the NLTK corpora required by the RAG tokenizer.

Run from the repository root after syncing the locked backend environment:

    backend/.venv/bin/python backend/scripts/setup_nltk.py
"""
import sys

import nltk

# punkt/punkt_tab -> word_tokenize; wordnet/omw-1.4 -> WordNetLemmatizer and
# the synonym lookup in rag/nlp/synonym.py
REQUIRED = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]


def main() -> int:
    failed = []
    for pkg in REQUIRED:
        if nltk.download(pkg, quiet=True):
            print(f"  ok      {pkg}")
        else:
            print(f"  FAILED  {pkg}")
            failed.append(pkg)

    try:
        from nltk import word_tokenize
        from nltk.corpus import wordnet

        word_tokenize("verify the tokenizer works")
        wordnet.synsets("width")
    except LookupError as exc:
        print(f"\nverification failed: {exc}")
        return 1

    if failed:
        print(f"\ncould not download: {', '.join(failed)}")
        return 1

    print("\nNLTK data ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
