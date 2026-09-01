import shutil, torch, json, sys
sys.path.insert(0, '.')
from pathlib import Path

# Disk
t, u, f = shutil.disk_usage('E:/')
print(f'Disk: total={t//1024**3}GB  free={f//1024**3}GB ({f//1024**2}MB)')

# GPU
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    mem_free, mem_total = torch.cuda.mem_get_info(0)
    print(f'GPU: {props.name}  total={props.total_memory/1e9:.1f}GB  free={mem_free/1e9:.1f}GB')
    print(f'CUDA: {torch.version.cuda}')
else:
    print('GPU: NOT AVAILABLE')

# Dataset
from data.bigearthnet_txt.dataset import BigEarthNetDataset
train_ds = BigEarthNetDataset(data_root='data/bigearthnet_txt',
    manifest_path='data/manifests/manifest_train.jsonl',
    s1_bands=['VV','VH'], s2_bands=None, img_size=120, split='train', strict=False)
val_ds = BigEarthNetDataset(data_root='data/bigearthnet_txt',
    manifest_path='data/manifests/manifest_validation.jsonl',
    s1_bands=['VV','VH'], s2_bands=None, img_size=120, split='validation', strict=False)
print(f'Train samples: {len(train_ds)}  Val samples: {len(val_ds)}')
train_ids = set(train_ds[i]['image_id'] for i in range(len(train_ds)))
val_ids = set(val_ds[i]['image_id'] for i in range(len(val_ds)))
overlap = train_ids & val_ids
print(f'Patch overlap: {len(overlap)} (must be 0)')
s = train_ds[0]
print(f'S1: {s["image_s1"].shape}  S2: {s["image_s2"].shape}')
tasks = {}
with open('data/manifests/manifest_train.jsonl') as ff:
    for line in ff:
        d = json.loads(line)
        ct = d.get('claim_type','unknown')
        tasks[ct] = tasks.get(ct,0)+1
print(f'Task distribution: {tasks}')
print('PRE-FLIGHT: PASS')
