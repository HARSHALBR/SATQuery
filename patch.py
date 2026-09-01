import re

with open('tests/test_evidence_comparator.py', 'r') as f:
    content = f.read()

helper = '''
def _change_quantification(eid: str = \"cq_1\", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type=\"change_quantification\",
        tool_version=\"mock-1.0\",
        value={\"changed_pixel_fraction\": 0.21, \"change_detected\": True},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov(\"change_statistics\"),
    )
'''
if 'def _change_quantification' not in content:
    content = content.replace('def _ndvi_decrease', helper + '\n\ndef _ndvi_decrease')

content = re.sub(r'evidence\s*=\s*\[_vlm_decrease\(([^)]*)\),\s*_ndvi_decrease\(([^)]*)\)\]', r'evidence = [_vlm_decrease(\g<1>), _ndvi_decrease(\g<2>), _change_quantification()]', content)
content = re.sub(r'evidence\s*=\s*\[_vlm_increase\(([^)]*)\),\s*_ndvi_increase\(([^)]*)\)\]', r'evidence = [_vlm_increase(\g<1>), _ndvi_increase(\g<2>), _change_quantification()]', content)
content = re.sub(r'\[vlm,\s*_ndbi_increase\(\)\]', r'[vlm, _ndbi_increase(), _change_quantification()]', content)
content = re.sub(r'\[vlm,\s*ndvi\]', r'[vlm, ndvi, _change_quantification()]', content)
content = re.sub(r'evidence\s*=\s*\[_vlm_decrease\(\),\s*_ndvi_increase\(([^)]*)\)\]', r'evidence = [_vlm_decrease(), _ndvi_increase(\g<1>), _change_quantification()]', content)

with open('tests/test_evidence_comparator.py', 'w') as f:
    f.write(content)

