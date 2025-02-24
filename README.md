# On the Adversarial Robustness of Visual-Language Chat Models

## Introdution

This is the official repository
of "On the Adversarial Robustness of Visual-Language Chat Models"

## Requirements

- Install required python packages:
```shell
python -m pip install -r requirements.py
```

## Training
Training commands are as follows.

* VQA:
```shell
python scripts/image_attack_vqa_target.py \
    --eps <noise budget> \
    --step_size <step size>   \
    --epoch <steps of PGD>          \
    --model <model choice>
```

* Jailbreaking:
```shell
python scripts/image_attack_bad_prompts.py \
    --eps <noise budget> \
    --step_size <step size>   \
    --epoch <steps of PGD>          \
    --model <model choice>
```

* Jailbreaking:
```shell
python scripts/image_attack_information.py \
    --eps <noise budget> \
    --step_size <step size>   \
    --epoch <steps of PGD>          \
    --model <model choice>
```

The parameter choices for the above commands are as follows:
- Noise budget `<--eps>`: `4` , `8`, `16`, `32`, `...`
- Steps of PGD `<--epoch>`: `100`, `200`, `300`, `400`, `...`
- Model choice `<--model>`: `minigpt4` , `minigpt5`, `mplug`, `mplug2`, `...`

The trained checkpoints will be saved at `scripts/experiments_<type>/<model>_eps=<args.eps>_epoch=<args.epoch>`.


## Acknowledgement
- Relevant VLMs:
  [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4),
  [MiniGPT-5](https://github.com/eric-ai-lab/MiniGPT-5),
  [mPlUG-owl](https://github.com/X-PLUG/mPLUG-Owl),
  [mPlUG-owl2](https://github.com/X-PLUG/mPLUG-Owl),
  [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA).