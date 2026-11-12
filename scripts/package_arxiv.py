"""Build a named PDF and minimal source archive; independently rebuild the archive."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader
from homefleet.paths import ROOT

TITLE = 'Adaptive Reliability Reserves for Residential Battery Fleets'
AUTHORS = [dict(name='Maruthi Vemula', affiliation='University of North Carolina at Chapel Hill', email='vemula@unc.edu'),
           dict(name='Neeraj Praneeth Gajula', affiliation='New York University', email='Npg8461@nyu.edu')]


def build(folder):
    try:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                       cwd=folder, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stdout.decode(errors='replace')[-10000:]) from error
    log = (folder / 'main.log').read_text(errors='replace')
    problems = [line for line in log.splitlines() if 'Overfull \\hbox' in line or
                'undefined' in line.lower() and ('citation' in line.lower() or 'reference' in line.lower()) or
                'multiply defined' in line.lower()]
    if problems: raise RuntimeError('Unresolved TeX issues:\n' + '\n'.join(problems))
    pdf = PdfReader(folder / 'main.pdf')
    text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    if any(token.lower() in text.lower() for token in ['anonymous author', 'anonymous.4open', 'under review', 'TODO', '[?]']):
        raise RuntimeError('Anonymous, review, or unresolved-placeholder text remains')
    # TeX kerning can make extractors emit "V emula" or join adjacent words.
    # Check exact non-whitespace identity on the title page, in author order.
    title_page = ''.join((pdf.pages[0].extract_text() or '').split()).casefold()
    compact = lambda value: ''.join(value.split()).casefold()
    for author in AUTHORS:
        if not all(compact(author[field]) in title_page for field in ['name', 'affiliation', 'email']):
            raise RuntimeError(f'Author identity missing from title page: {author["name"]}')
    if [title_page.index(compact(a['name'])) for a in AUTHORS] != sorted(title_page.index(compact(a['name'])) for a in AUTHORS):
        raise RuntimeError('Title-page author order differs from submission metadata')
    return pdf, text


def plain_abstract(paper):
    source = (paper / 'main.tex').read_text()
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, flags=re.S).group(1)
    numbers = (paper / 'numbers.tex').read_text()
    macros = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}', numbers))
    for name in sorted(macros, key=len, reverse=True): abstract = abstract.replace('\\' + name, macros[name])
    abstract = abstract.replace(r'\%', ' percent').replace(r'\$', 'USD ').replace('\\ ', ' ').replace('--', '-')
    if re.search(r'\\[A-Za-z]', abstract): raise ValueError('Unexpanded command in submission abstract')
    abstract = ' '.join(abstract.split())
    if len(abstract) > 1920 or not abstract.isascii():
        raise ValueError('arXiv metadata requires an ASCII abstract of at most 1920 characters')
    return abstract


def main():
    paper = ROOT / 'paper'
    target = ROOT / 'release/arxiv'
    target.mkdir(parents=True, exist_ok=True)
    original, original_text = build(paper)
    # Explicit source traversal includes only files used by this manuscript.
    required = set()
    pending = [Path('main.tex')]
    while pending:
        relative = pending.pop()
        if relative in required: continue
        required.add(relative)
        text = (paper / relative).read_text()
        for name in re.findall(r'\\input\{([^}]+)\}', text):
            p = Path(name); pending.append(p if p.suffix else p.with_suffix('.tex'))
        for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            p = Path(name); required.add(p if p.suffix else p.with_suffix('.pdf'))
    required.update(map(Path, ['neurips_2026.sty', 'refs.bib', 'main.bbl']))
    with tempfile.TemporaryDirectory(prefix='homefleet-arxiv-') as directory:
        clean = Path(directory)
        for relative in sorted(required):
            destination = clean / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paper / relative, destination)
        rebuilt, rebuilt_text = build(clean)
        if len(rebuilt.pages) != len(original.pages) or rebuilt_text != original_text:
            raise RuntimeError('The clean source rebuild does not match manuscript text/pages')
        source_zip = target / 'homefleet-arxiv-source.zip'
        with zipfile.ZipFile(source_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in sorted(required): archive.write(clean / relative, str(relative))
    pdf_target = target / 'adaptive-reliability-reserves.pdf'
    shutil.copy2(paper / 'main.pdf', pdf_target)
    output = ROOT / 'output/pdf'
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_target, output / pdf_target.name)
    abstract = plain_abstract(paper)
    (target / 'abstract.txt').write_text(abstract + '\n')
    metadata = dict(title=TITLE, authors=AUTHORS,
                    authors_field=', '.join(a['name'] + ' (' + a['affiliation'] + ')' for a in AUTHORS), abstract=abstract,
                    suggested_primary_category='eess.SY', suggested_cross_list=['cs.LG'],
                    suggested_paper_license=dict(name='arXiv.org perpetual, non-exclusive license 1.0',
                        url='https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html'),
                    license_selected_in_arxiv=False,
                    comments=f'{len(original.pages)} pages, 5 figures; code and artifacts: https://github.com/MaruthiV/homefleet',
                    journal_reference=None, arxiv_identifier=None, submitted=False)
    (target / 'submission-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    report = dict(title=TITLE, pages=len(original.pages), clean_rebuild_matches_text=True,
                  pdf_sha256=hashlib.sha256(pdf_target.read_bytes()).hexdigest(),
                  source_zip_sha256=hashlib.sha256(source_zip.read_bytes()).hexdigest(),
                  source_files=[dict(path=str(p), sha256=hashlib.sha256((paper / p).read_bytes()).hexdigest()) for p in sorted(required)])
    (target / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / 'artifacts/arxiv_build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(pdf_target, source_zip, f'{len(original.pages)} pages', sep='\n')


if __name__ == '__main__':
    main()
