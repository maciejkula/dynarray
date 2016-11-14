import os
import re


import numpy as np  # NOQA


def test_examples():

    readme_path = os.path.join(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..'
    ), 'readme.md')

    with open(readme_path, 'rb') as readme_file:
        readme_text = readme_file.read()
        examples = re.findall('```[^`]*```', readme_text, flags=re.DOTALL)

        for number, example in enumerate(examples):
            source = example.replace('`', '').replace('python', '')
            code = compile(source, 'example_{}'.format(number), 'exec')
            exec(code)
