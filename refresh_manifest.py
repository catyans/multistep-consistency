"""Refresh distribution checksums after intentionally editing repository files."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
IGNORED = {'.git', '__pycache__', 'outputs', '.venv', '.venv-official'}


def package_files():
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in IGNORED or part.startswith('.venv') for part in rel.parts):
            continue
        if path.name in {'SHA256SUMS.json', '.DS_Store', 'installed-environment.txt', 'classify_image_graph_def.pb'} or path.name.startswith('._') or path.suffix == '.pyc':
            continue
        yield rel, path


if __name__ == '__main__':
    hashes = {rel.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for rel, path in package_files()}
    (ROOT / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(f'Hashed {len(hashes)} distribution files.')
