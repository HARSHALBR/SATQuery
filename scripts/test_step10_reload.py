import sys, json, torch
from pathlib import Path
sys.path.insert(0, '.')

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.lora import load_lora_checkpoint
from data.bigearthnet_txt.dataset import BigEarthNetDataset
from transformers import AutoTokenizer

print('=== STEP 10 CHECKPOINT RELOAD VERIFICATION ===')
ckpt_dir = Path('checkpoints/pretrained_lora/best')
assert ckpt_dir.exists(), f'Checkpoint directory {ckpt_dir} does not exist!'
assert (ckpt_dir / 'modality_encoders.pt').exists(), 'modality_encoders.pt missing'
assert (ckpt_dir / 'modality_projections.pt').exists(), 'modality_projections.pt missing'
assert (ckpt_dir / 'adapter').exists(), 'adapter directory missing'
assert (ckpt_dir / 'metrics.json').exists(), 'metrics.json missing'

with open(ckpt_dir / 'metrics.json') as f:
    m = json.load(f)
print(f'Loaded checkpoint metadata: {m}')

print('Reconstructing RSInternVL model from modular checkpoint...')
cfg = RSInternVLConfig(
    model_id='OpenGVLab/InternVL3-1B',
    pretrained_backbone=True,
    img_size=120,
    s1_channels=2,
    s1_hidden_dim=512,
    s2_channels=10,
    s2_hidden_dim=768,
    projection_hidden_dim=1024,
    freeze_llm=True,
)
model = load_lora_checkpoint(
    checkpoint_dir=ckpt_dir,
    config_override=cfg,
    device='cpu',
    is_trainable=False,
)
model.eval()
print('Model successfully reconstructed and placed on CPU.')

# Test inference on 1 validation sample
val_ds = BigEarthNetDataset(
    data_root='data/bigearthnet_txt',
    manifest_path='data/manifests/manifest_validation.jsonl',
    s1_bands=['VV', 'VH'],
    s2_bands=None,
    img_size=120,
    split='validation',
    strict=False,
)
sample = val_ds[0]
s1 = sample['image_s1'].unsqueeze(0)
s2 = sample['image_s2'].unsqueeze(0)
query = sample['text']
target = sample['target_text']

tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-1B', trust_remote_code=True)
prompt = f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)])
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    gen_tokens, _ = model.generate(
        image_s1=s1,
        image_s2=s2,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        do_sample=False,
    )
gen_text = tokenizer.decode(gen_tokens[0].tolist(), skip_special_tokens=True).strip()
print(f'Query:     {query}')
print(f'Target:    {target}')
print(f'Generated: {gen_text}')
assert len(gen_text) > 0, 'Generated text is empty!'
print('RELOAD VERIFICATION: PASS')
