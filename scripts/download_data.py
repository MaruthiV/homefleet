import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import requests
from homefleet.paths import ROOT


def download(group, destination, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    selected = [a for a in manifest['assets'] if group == 'all' or a['group'] == group]
    if not selected:
        raise ValueError(f'no released assets for {group}')
    cache = destination / 'release/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    for asset in selected:
        path = cache / asset['name']
        valid = False
        if path.exists():
            with path.open('rb') as handle:
                valid = hashlib.file_digest(handle, 'sha256').hexdigest() == asset['sha256']
        if not valid:
            partial = path.with_suffix('.part')
            digest = hashlib.sha256()
            with requests.get(asset['url'], stream=True, timeout=300) as response:
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk); digest.update(chunk)
            if digest.hexdigest() != asset['sha256']:
                raise ValueError(f'checksum mismatch: {asset["name"]}')
            partial.replace(path)
        with tarfile.open(path, 'r:gz') as archive:
            members = archive.getmembers()
            for member in members:
                parts = Path(member.name).parts
                if not parts or parts[0] not in {'data', 'experiments', 'artifacts'}:
                    raise ValueError(f'unexpected archive member: {member.name}')
            archive.extractall(destination, members=members, filter='data')
        print('Verified and extracted', asset['name'], flush=True)


def main():
    ap = argparse.ArgumentParser(description='Download hash-verified frozen research artifacts.')
    ap.add_argument('--group', choices=['bundles', 'raw', 'checkpoints', 'results', 'all'], default='bundles')
    ap.add_argument('--destination', type=Path, default=ROOT)
    ap.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/release_manifest.json')
    a = ap.parse_args()
    download(a.group, a.destination.resolve(), a.manifest)


if __name__ == '__main__':
    main()
