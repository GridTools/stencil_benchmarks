import subprocess


def format_code(code):
    result = subprocess.run('clang-format',
                            input=code,
                            encoding='ascii',
                            stdout=subprocess.PIPE,
                            check=True)
    return result.stdout
