
import os, time
from dotenv import load_dotenv
load_dotenv()
from tools.vlm.gemini_client import GeminiVLMClient
from schemas.query import ObservationRole
from agents.real_runner import RealToolRunner
from tests.test_vlm_integration import _build_obs
from backend.services.orchestrator import SATQueryOrchestrator

client = GeminiVLMClient()

def run_case(name, obs1, obs2, query, base_claim):
    print(f'\n================================')
    print(f'=== Case {name} ===')
    print(f'================================')
    
    obs = [_build_obs(obs1, ObservationRole.T1, '2021-01-01', 'id1'), _build_obs(obs2, ObservationRole.T2, '2021-01-02', 'id2')]
    runner = RealToolRunner(observations=obs, vlm_client=client)
    runner.execute('validate_inputs', {})
    
    ndvi_res = runner.execute('ndvi_delta', {'input_ids': ['id1', 'id2']})
    rs_res = runner.execute('change_statistics', {'delta_map': ndvi_res.output['delta_map'], 'valid_mask': ndvi_res.output['valid_mask'], 'input_ids': ['id1', 'id2']})
    
    t0 = time.time()
    vlm_res = runner.execute('run_rs_vlm', {'query': query})
    t1 = time.time()
    latency = t1 - t0
    
    print('API Call Success:', vlm_res.status.name == 'SUCCESS')
    if vlm_res.status.name != 'SUCCESS':
        print('Error:', vlm_res.error)
        return
        
    ev = vlm_res.output['evidence']
    print('Model Used:', ev.tool_version)
    print('VLM Claim:', ev.value.get('claim'))
    print('Confidence:', ev.value.get('confidence'))
    print('Reasoning:', ev.value.get('reasoning'))
    print(f'Latency: {latency:.2f} seconds')
    print('Provenance Tool:', ev.provenance.tool)
    print('Input IDs:', ev.provenance.input_ids)
    
    orch = SATQueryOrchestrator(runner=runner)
    comp_result = orch.comparator.compare(base_claim, [ndvi_res.output['evidence'], rs_res.output['evidence'], ev])
    
    print('Comparator Status:', comp_result.status.name)
    print('Supporting IDs:', comp_result.supporting_evidence)
    print('Conflicting IDs:', comp_result.conflicting_evidence)

run_case('A', 'datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A', 'datasets/golden_fixtures/raw/S2B_10TFK_20211001_0_L2A', 'Has vegetation decreased?', 'vegetation_decrease')
run_case('B', 'datasets/golden_fixtures/raw/S2B_10TDL_20210706_1_L2A', 'datasets/golden_fixtures/raw/S2B_10TDL_20211014_1_L2A', 'Has vegetation decreased?', 'vegetation_decrease')
run_case('C', 'datasets/golden_fixtures/raw/S2A_10SGF_20210705_2_L2A', 'datasets/golden_fixtures/raw/S2A_10SGF_20211013_0_L2A', 'Has there been permanent deforestation?', 'vegetation_decrease')
