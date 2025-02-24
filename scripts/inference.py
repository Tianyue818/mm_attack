import os
import sys
import argparse

package = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(package)

import torch

from mmr.models import MODEL_LIST, get_model


def parse(argv):
    args = {
        ('model', ): {
            'type': str,
            'choices': MODEL_LIST,
            'help': 'Model to use for inference.',
        },
        ('question', ): {
            'type': str,
            'help': 'Question to ask the model.'
        },
        ('-i', '--image'): {
            'type': str, 'default': None,
            'help': 'A path to the image to use as context for the question.'
        },
        ('-nr', '--num-responses'): {
            'type': int, 'default': 1,
            'help': 'The number of responses to generate.'
        },
        ('--download', ): {
            'action': 'store_true',
            'help': 'Download model if not cached.',
        }
    }
    parser = argparse.ArgumentParser()
    for arg, kwargs in args.items():
        parser.add_argument(*arg, **kwargs)
    return parser.parse_args(argv[1:])


def main(args):
    torch.set_default_device('cuda')
    model = get_model(
        args.model,
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        # use_cache=False,
        # device_map='auto',
        local_files_only=not args.download,
        resume_download=not args.download,
        # low_cpu_mem_usage=True,
    )
    image = None
    if args.image is not None:
        image = model.raw_image_to_tensor(args.image, device='cuda')
    question = args.question
    nr = args.num_responses
    prompts = model.format_prompts([question] * nr, None, [image] * nr)
    text = model.generate(prompts)
    for i, t in enumerate(text):
        print(f'------------ {i} ------------')
        print(t)


if __name__ == '__main__':
    main(parse(sys.argv))
