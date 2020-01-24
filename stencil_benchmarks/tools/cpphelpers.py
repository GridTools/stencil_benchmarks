import warnings
import subprocess


def format_code(code):
    try:
        code = subprocess.run('clang-format',
                              input=code,
                              encoding='ascii',
                              stdout=subprocess.PIPE,
                              check=True).stdout
    except FileNotFoundError:
        warnings.warn('C++ code not formatted: could not find clang-format')
    return ''.join(f'{num + 1:4} {line}\n'
                   for num, line in enumerate(code.split('\n')))
