# Driving methods

Set the source, checkpoint and interpreter paths for your selected methods in the shell or in `resource.env`. [Example.](../configs/model-environment.example.json)

| Method ID | Required source and configuration |
| --- | --- |
| `TM` | Traffic Manager is included with CARLA 0.9.16. |
| `IDM` | The included IDM controller uses CARLA BasicAgent for route tracking. |
| `LEAD` | Set `LEAD_SOURCE` to the checkout containing `lead/` and its submodules. Set `LEAD_WEIGHTS` to `tfv6_resnet34/`, with `config.json` and `model_0030_0.pth` from the same directory. [LEAD source](https://github.com/kesai-labs/lead/tree/cvpr2026) · [ResNet-34 weights](https://huggingface.co/ln2697/tfv6/tree/main/tfv6_resnet34) |
| `OneVL` | Set `ONEVL_WEIGHTS` to the directory containing the model and processor files from the OneVL Impromptu checkpoint. [Checkpoint](https://huggingface.co/xiaomi-research/OneVL_Impromptu) |
| `Alpamayo` | Install `alpamayo1_5` in the inference environment. Set `ALPAMAYO_WEIGHTS` to the parent of the three directories below. [Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5) |
| `UniAD` | Install the UniAD model and CUDA operators from [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) in `UNIAD_PYTHON`. Set `UNIAD_CONFIG` to `adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py` in that repository and keep the accompanying `configs/_base_/` directory. Set `UNIAD_WEIGHTS` to the downloaded `uniad_base_b2d.pth`. Copy `data/others/b2d_motion_anchor_infos_mode6.pkl` from the checkout into the checkpoint directory, or set `visual_policy.motion_anchor` to the anchor file's absolute path. [UniAD-Base checkpoint](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/uniad_base_b2d.pth) |

### LEAD and UniAD

Set `LEAD_PYTHON` or `UNIAD_PYTHON` to the method's Python 3.10 interpreter. Both methods use the CARLA 0.9.16 client and matching navigation agents. LEAD's upstream environment pins CARLA 0.9.15, so keep the benchmark's 0.9.16 client when installing LEAD. In each selected environment, install the runtime dependencies and PyTorch for CUDA 11.8.

```bash
python -m pip install -r requirements/runtime.txt
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

For UniAD, set `CUDA_HOME` to the CUDA 11.8 toolkit and build the operators from the Bench2DriveZoo checkout containing `setup.py` and `mmcv/`.

```bash
"$UNIAD_PYTHON" -m pip install ninja packaging wheel
"$UNIAD_PYTHON" -m pip install --no-build-isolation /path/to/Bench2DriveZoo
"$UNIAD_PYTHON" -c "from mmcv.models.detectors.uniad_e2e import UniAD"
```

### OneVL and Alpamayo

Set `VLM_CARLA_PYTHON` to the Python 3.10 CARLA environment and `VLM_INFERENCE_PYTHON` to a separate Python 3.12 environment with PyTorch 2.8.0 (CUDA 12.8), torchvision 0.23.0 and Transformers 4.57.1.

For Alpamayo, run the downloads below with a Hugging Face account that has access to [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B).

```bash
hf download nvidia/Alpamayo-1.5-10B --local-dir "$ALPAMAYO_WEIGHTS/model"
hf download nvidia/Cosmos-Reason2-8B --include '*.json' '*.txt' '*.jinja' --local-dir "$ALPAMAYO_WEIGHTS/base-config"
hf download Qwen/Qwen3-VL-2B-Instruct --include '*.json' '*.txt' '*.jinja' --local-dir "$ALPAMAYO_WEIGHTS/processor"
```
