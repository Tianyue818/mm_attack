import os
import sys
import pandas as pd
root = os.path.dirname(__file__)
package = os.path.join(root, '..')
sys.path.append(package)

import torch
import torchvision as tv
import argparse

from mmr.models import get_model
from mmr.attack import average_levenshtein_distance as distance

from pgd import pgd_attack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--eps", type=int, default=16)
    parser.add_argument("--step_size", type=float, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--model", type=str, default='minigpt4')
    args = parser.parse_args()

    torch.set_default_device('cuda')
    test_image_path = './tests/test_ifh.jpg'
    model_name = args.model
    data_path = os.path.join(root, 'experiments_information')
    save_model_name = f'{model_name}_eps={args.eps}_epoch={args.epoch}'
    save_path = os.path.join(data_path, save_model_name)
    os.makedirs(save_path, exist_ok=True)
    test_model = get_model(
        model_name,
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        local_files_only=True,
        # resume_download=True,
    )


    prompt_path = './dataset/Information/secret_information.txt'
    
    prompt_data = open(prompt_path, "r")
    prompt_targets = []
    for line in prompt_data:
        prompt_targets.append(line)
    prompt_data.close()

    print(prompt_targets)
    question = 'What is depicted in the figure? Qgac'
    test_image = test_model.raw_image_to_tensor(test_image_path, device='cuda')

    print('Adversarial Attack of information hiding')


    bs = args.bs
    for ep in range(args.begin, args.end, bs):
        print(prompt_targets[ep:ep + bs])
        def early_stop(o):
            dist = distance(o,  prompt_targets[ep:ep+len(o)])
            stop = dist < 0.005
            print(f'{dist=}, {stop=}')
            return stop
        tt_dict = pgd_attack(
            test_model, [question]*bs, [test_image] * bs, prompt_targets[ep:ep + bs],
            args.eps / 255, args.step_size / 255, generate=1, steps=args.epoch, early_stop=early_stop)
        adv_image = tt_dict['attack_images']
        # print(adv_image)

        adv_prompts = test_model.format_prompts([question]*bs, None, adv_image)
        text = test_model.generate(adv_prompts)

        for i, t in enumerate(text):
            print(f'------------ {i} ---------------')
            print(t)
        print(f'Saving adversarial image to "adv_ih_{ep}.png".')
        for i in range(bs):
            adv_image_path = os.path.join(save_path, f'adv_ih_{i+ep}.png')
            tv.utils.save_image(adv_image[i], adv_image_path)

if __name__ == "__main__":
    main()
