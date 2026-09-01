import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app
from schemas.response import EvidenceStatus, FinalResponse

client = TestClient(app)

def test_real_runner_injection():
    with patch('backend.routes.analyze.RealToolRunner') as MockRunner:
        with patch('backend.routes.analyze.SATQueryOrchestrator') as MockOrchestrator:
            mock_orch_instance = MockOrchestrator.return_value
            mock_orch_instance.analyze.return_value = FinalResponse(
                trace_id='dummy', task='dummy', answer='dummy', status=EvidenceStatus.SUPPORTED, evidence=[]
            )
            with patch.dict(os.environ, {'MOCK_RS_TOOLS': 'false'}):
                payload = {'query': 'Test', 'observations': []}
                client.post('/api/v1/analyze', json=payload)
                assert MockRunner.called
                kwargs = MockOrchestrator.call_args.kwargs
                assert kwargs['runner'] == MockRunner.return_value

def test_mock_runner_injection():
    with patch('backend.routes.analyze.RealToolRunner') as MockRunner:
        with patch('backend.routes.analyze.SATQueryOrchestrator') as MockOrchestrator:
            mock_orch_instance = MockOrchestrator.return_value
            mock_orch_instance.analyze.return_value = FinalResponse(
                trace_id='dummy', task='dummy', answer='dummy', status=EvidenceStatus.SUPPORTED, evidence=[]
            )
            with patch.dict(os.environ, {'MOCK_RS_TOOLS': 'true'}):
                payload = {'query': 'Test', 'observations': []}
                client.post('/api/v1/analyze', json=payload)
                assert not MockRunner.called
                kwargs = MockOrchestrator.call_args.kwargs
                assert kwargs['runner'] is None

def test_missing_gemini_credentials_fail_safely():
    with patch.dict(os.environ, {'MOCK_RS_TOOLS': 'false', 'MOCK_VLM': 'false', 'GEMINI_API_KEY': ''}):
        payload = {
            'query': 'Has vegetation decreased?',
            'observations': [
                {
                    'observation_id': 'a1',
                    'image_path': 'datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A',
                    'role': 't1',
                    'metadata': {'modality': 'optical', 'bands': ['red', 'nir', 'scl']}
                },
                {
                    'observation_id': 'a2',
                    'image_path': 'datasets/golden_fixtures/raw/S2A_10TFK_20211026_0_L2A',
                    'role': 't2',
                    'metadata': {'modality': 'optical', 'bands': ['red', 'nir', 'scl']}
                }
            ]
        }
        response = client.post('/api/v1/analyze', json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == EvidenceStatus.INSUFFICIENT.value
        assert 'missing' in data['limitations'][0].lower() or 'error' in data['limitations'][0].lower()
