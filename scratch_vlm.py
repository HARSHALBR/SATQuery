
import os
from dotenv import load_dotenv
load_dotenv()
from tools.vlm.gemini_client import GeminiVLMClient
from schemas.query import ObservationRole
from agents.real_runner import RealToolRunner
from tests.test_vlm_integration import _build_obs

client = GeminiVLMClient()

def run_case(name, obs1, obs2, query):
    print(f'\n=== Case {name} ===')
    obs = [_build_obs(obs1, ObservationRole.T1, '2021-01-01', 'id1'), _build_obs(obs2, ObservationRole.T2, '2021-01-02', 'id2')]
    runner = RealToolRunner(observations=obs, vlm_client=client)
    runner.execute('validate_inputs', {})
    runner.execute('ndvi_delta', {'input_ids': ['id1', 'id2']})
    vlm_res = runner.execute('run_rs_vlm', {'query': query})
    print('Status:', vlm_res.status)
    print('Error:', vlm_res.error)

run_case('A', 'datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A', 'datasets/golden_fixtures/raw/S2B_10TFK_20211001_0_L2A', 'Has vegetation decreased?')
run_case('B', 'datasets/golden_fixtures/raw/S2B_10TDL_20210706_1_L2A', 'datasets/golden_fixtures/raw/S2B_10TDL_20211014_1_L2A', 'Has vegetation decreased?')
run_case('C', 'datasets/golden_fixtures/raw/S2A_10SGF_20210705_2_L2A', 'datasets/golden_fixtures/raw/S2A_10SGF_20211013_0_L2A', 'Has there been permanent deforestation?')
