# Install

Follow all the standard installation setting, makesure diffusers>=0.14

1. Download the base model 
```
python3 scripts/download_pipeline.py --token ACCESS_TOKEN
```
2. Build the AIT modules by running `compile.py`.
```
python3 scripts/compile.py
```
3. Run Inference

```
python3 scripts/demo_controlnet.py
```
