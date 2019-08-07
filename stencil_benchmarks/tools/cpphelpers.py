import subprocess


def format_code(code):
    result = subprocess.run('clang-format',
                            input=code,
                            encoding='ascii',
                            stdout=subprocess.PIPE,
                            check=True)
    return result.stdout


def loop(index, loop_range, body, index_type='int'):
    if loop_range.step > 0:
        comp = '<'
    else:
        comp = '>'
    step = loop_range.step if loop_range.step is not None else 1
    return format_code(f'for ({index_type} {index} = {loop_range.start}; '
                       f'{index} {comp} {loop_range.stop}; '
                       f'{index} += {step}) {{'
                       f'{body}'
                       '}')


def nested_loops(indices, ranges, body, index_type='int'):
    code = body
    for index, loop_range in zip(reversed(indices), reversed(ranges)):
        code = loop(index, loop_range, code, index_type=index_type)
    return format_code(code)
