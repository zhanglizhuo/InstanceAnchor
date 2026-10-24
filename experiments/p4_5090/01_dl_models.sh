#!/bin/bash
# experiments/p4_5090/01_dl_models.sh — download 5 paper backbones to HF cache
# Run on 5090: nohup setsid bash 01_dl_models.sh > logs/dl_models.log 2>&1 < /dev/null &
export HF_ENDPOINT=https://hf-mirror.com
VENV=~/ia_p4/venv/bin/python
$VENV -c "
import open_clip, torch, time
targets = [('ViT-L-14','openai'),('ViT-L-14','laion2b_s32b_b82k'),
           ('EVA02-L-14','merged2b_s4b_b131k'),('ViT-L-14','dfn2b')]
targets.append(('hf-hub:google/siglip2-large-patch16-256', None))
for spec in targets:
    t0=time.time()
    try:
        if spec[1] is None:
            m,_,t = open_clip.create_model_and_transforms(spec[0], device='cpu')
        else:
            m,_,t = open_clip.create_model_and_transforms(spec[0], pretrained=spec[1], device='cpu')
        print('OK', spec, round(time.time()-t0,1), 's', flush=True)
        del m
    except Exception as e:
        print('FAIL', spec, str(e)[:150], flush=True)
"
echo ALL_MODELS_DONE
