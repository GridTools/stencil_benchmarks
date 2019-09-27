import subprocess


def format_code(code):
    code = subprocess.run('clang-format',
                          input=code,
                          encoding='ascii',
                          stdout=subprocess.PIPE,
                          check=True).stdout
    return ''.join(f'{num + 1:4} {line}\n'
                   for num, line in enumerate(code.split('\n')))
