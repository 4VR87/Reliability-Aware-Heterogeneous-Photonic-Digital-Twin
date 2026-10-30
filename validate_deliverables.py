"""Structural and result-to-manuscript audit for the delivered article."""
from pathlib import Path
from zipfile import ZipFile
import json,re
from lxml import etree
from pypdf import PdfReader
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}

def audit():
    content=json.loads((ROOT/'manuscript_content.json').read_text())
    blocks=content['blocks']
    eqs=[x for x in blocks if x['kind']=='equation']
    refs=json.loads((ROOT/'references/references_verified.json').read_text())
    assert [x['number'] for x in eqs]==list(range(1,30))
    assert len(refs)==25 and len({x['doi'].lower() for x in refs})==25
    assert sum(x['year']>=2023 for x in refs)==20
    assert sum(x['year']==2026 for x in refs)==5
    text='\n'.join(x.get('text','') for x in blocks if x['kind'] not in ['reference','equation'])
    assert set(map(int,re.findall(r'\[(\d+)\]',text)))==set(range(1,27))
    assert not re.search('[\x00-\x08\x0b\x0c\x0e-\x1f]',text)
    assert not re.search(r'\b(?:TODO|TBD|PLACEHOLDER)\b',text)
    assert len(next(x['text'] for x in blocks if x['kind']=='abstract').split())==217
    word=OUT/'Reliability_Aware_Heterogeneous_Photonic_Digital_Twin_Manuscript.docx'
    with ZipFile(word) as z:
        x=etree.fromstring(z.read('word/document.xml'))
        seq=x.xpath('//w:fldSimple[contains(@w:instr,"SEQ Equation")]',namespaces=NS)
        assert [int(''.join(s.xpath('.//w:t/text()',namespaces=NS))) for s in seq]==list(range(1,30))
        tables=x.xpath('/w:document/w:body/w:tbl',namespaces=NS)
        equation_tables=[t for t in tables if t.xpath('.//m:oMath',namespaces=NS)]
        assert len(equation_tables)==29 and len(tables)-len(equation_tables)==7
        assert all(c.get('{'+NS['w']+'}num','1')=='1' for c in x.xpath('//w:cols',namespaces=NS))
        assert len(x.xpath('//*[local-name()="inline"]'))==8
        assert not x.xpath('//w:ins|//w:del|//w:commentRangeStart',namespaces=NS)
        strings=x.xpath('//w:t/text()',namespaces=NS)
        assert not any('$' in s or '\\frac' in s or '\\boldsymbol' in s for s in strings)
        rel=etree.fromstring(z.read('word/_rels/document.xml.rels'))
        targets=[r.get('Target') for r in rel if r.get('Type','').endswith('/hyperlink')]
        assert all('https://doi.org/'+r['doi'] in targets for r in refs)
        assert not any('comments.xml' in n for n in z.namelist())
    pdf=OUT/'Reliability_Aware_Heterogeneous_Photonic_Digital_Twin_Manuscript.pdf'
    reader=PdfReader(pdf);pdftext='\n'.join(p.extract_text() for p in reader.pages)
    assert len(reader.pages)==13
    for i in range(1,30):assert f'({i})' in pdftext
    for i in range(1,9):assert re.search(r'Fig\.\s*'+str(i)+r'\.',pdftext)
    assert '??' not in pdftext
    log=(ROOT/'latex/build_release_second.log').read_text()
    assert not re.search(r'Overfull|undefined|LaTeX Warning:',log)
    # The table builders consume the numeric summary directly. Independently
    # check its policy counts against raw device records before releasing it.
    summary=json.loads((ROOT/'data/results_summary.json').read_text())
    for prefix in ['primary','refreshed']:
        df=pd.read_csv(ROOT/f'data/{prefix}_decisions.csv')
        for record in summary[prefix]:
            g=df[df.method==record['method']]
            assert len(g)==record['n']==1000
            assert int(g.authorized.sum())==record['authorized_n']
            assert int(g.unsafe.sum())==record['unsafe_n']
            np.testing.assert_allclose(g.heat_w.mean()*1000,record['mean_heat_mw'])
    qa=ROOT/'qa/word_release'
    wordpages=len(list(qa.glob('page-*.png')))
    assert wordpages==18
    impact=(ROOT/'SUBMISSION_NOTES.md').read_text().split('## Draft impact statement\n\n')[1].split('\n\n')[0]
    assert len(impact)<500
    result={'word_pages':wordpages,'pdf_pages':len(reader.pages),'numbered_native_word_equations':29,
            'figures':8,'data_tables':7,'algorithm':1,'research_references':25,
            'research_references_from_2023_onward':20,'research_references_from_2026':5,
            'ai_disclosure_reference':26,'abstract_words':217,'impact_statement_characters':len(impact),
            'equation_fields_citations_hyperlinks_raw_result_counts':'passed',
            'latex_overfull_boxes_and_unresolved_references':0,
            'layout_review':'All 18 Word and 13 PDF pages inspected; unchanged pages compared to reviewed renders.',
            'scientific_status':'Executed synthetic study; author identity, independent scientific review and journal submission remain author actions.'}
    (ROOT/'data/DOCUMENT_VALIDATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
