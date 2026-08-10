"""Allow ``python -m trans_novel`` and PyInstaller to run the CLI."""

from trans_novel.cli import main

if __name__ == "__main__":
    main()
