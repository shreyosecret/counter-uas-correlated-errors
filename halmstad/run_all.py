"""Rebuild every vote file from scratch (the Colab runtime was recycled), then run the audit."""
import subprocess, os, json
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh('pip install -q mcos-decoder open_clip_torch "transformers>=4.49" accelerate statsmodels opencv-python-headless openpyxl')
if not os.path.exists('hh'):
    for c in ["git clone -q --filter=blob:none --no-checkout https://github.com/DroneDetectionThesis/Drone-detection-dataset.git hh",
              "git -C hh sparse-checkout init --no-cone",
              "git -C hh sparse-checkout set 'Data/Video_V/V_DRONE_*' 'Data/Video_V/V_BIRD_*' 'Data/Video_IR/IR_DRONE_*' 'Data/Video_IR/IR_BIRD_*' 'Data/Video_dataset_description.xlsx'",
              "git -C hh checkout -q"]:
        sh(c)
import numpy as np, pandas as pd
from halmstad_lib import extract_crops
S = ['V', 'IR']
for s in S:
    if not os.path.exists(f'crops_{s}/meta.json'):
        t, c = extract_crops('hh/Data', s, out_dir=f'crops_{s}'); print(s, len(t), 'clips', len(c), 'crops', flush=True)
for s in S:
    if not os.path.exists(f'clip_{s}.csv'):
        sh(f'python dvb_tau_pipeline.py query --crops crops_{s} --backend clip --out clip_{s}.csv > /dev/null 2>&1'); print('clip done', s, flush=True)
import torch
from PIL import Image
if not all(os.path.exists(f'probe_{s}.csv') for s in S):
    import open_clip
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    cm, _, prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device='cuda')
    for s in S:
        m = json.load(open(f'crops_{s}/meta.json')); X = []
        with torch.no_grad():
            for i in range(0, len(m), 64):
                b = torch.stack([prep(Image.open(f'crops_{s}/' + r['file']).convert('RGB')) for r in m[i:i + 64]]).to('cuda')
                e = cm.encode_image(b); X.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        X = np.vstack(X); y = np.array([int(r['truth'] == 'drone') for r in m]); g = np.array([r['track'] for r in m])
        p = cross_val_predict(LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000), X, y, cv=GroupKFold(5), groups=g)
        d = pd.DataFrame(m); d['vote'] = p; d['truth'] = y; d.to_csv(f'probe_{s}.csv', index=False); print('probe done', s, flush=True)
    del cm; torch.cuda.empty_cache()
if not all(os.path.exists(f'qwen2_{s}.csv') for s in S):
    from dvb_tau_pipeline import QwenBackend
    qb = QwenBackend()
    Q2 = 'Is there a drone (a small quadcopter or other unmanned aircraft) in this image? Answer yes or no.'
    for s in S:
        m = json.load(open(f'crops_{s}/meta.json')); rows = []
        q = Q2 if s == 'V' else Q2.replace('this image', 'this thermal infrared image')
        for r in m:
            pil = Image.open(f'crops_{s}/' + r['file']).convert('RGB')
            msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}]
            text = qb.proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = qb.proc(text=[text], images=[pil], return_tensors='pt').to(qb.model.device)
            out = qb.model.generate(**inp, max_new_tokens=3, do_sample=False)
            ans = qb.proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip().lower()
            rows.append(dict(file=r['file'], track=r['track'], vote=int(ans.startswith('yes')), truth=int(r['truth'] == 'drone')))
        pd.DataFrame(rows).to_csv(f'qwen2_{s}.csv', index=False); print('qwen done', s, flush=True)
for s in S:
    sh(f'python audit_real.py hh/Data {s} > audit_{s}.txt 2>&1; cat audit_{s}.txt')
print('ALL DONE', flush=True)
