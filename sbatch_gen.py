#!/usr/bin/env python

from __future__ import print_function
import itertools
import sys
import re
import hashlib

def var_range(s):
    # range
    p = re.compile(r'(\d+)-(\d+)(:([\\+-\\*/]?)(\d+))?')
    m = p.match(s)
    if m:
        first = int(m.group(1))
        last = int(m.group(2))
        if m.group(3) is not None:
            op = '+' if m.group(4) is None else m.group(4)
            step = int(m.group(5))
        else:
            op = '+' if first <= last else '-'
            step = 1
        def stepop(x):
            while True:
                yield x
                if op == '+':
                    x += step
                if op == '-':
                    x -= step
                if op == '*':
                    x *= step
                if op == '/':
                    x //= step
        return list(itertools.takewhile(lambda x: x <= last, stepop(first)))
    # comma-separated list
    p = re.compile(r'[^,]+')
    return p.findall(s)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} [APPLICATION] [APPLICATION ARGUMENTS]...'.format(sys.argv[0]))
        print()
        print('This script generates a slurm array job script for starting the given application with the given arguments.')
        print('The result is printed to stdout. Arguments given in brackets are interpreted as ranges of values.')
        print()
        print('Supported ranges:')
        print('  [id]          will be replaced by $SLURM_ARRAY_TASK_ID, i.e. a unique linear job index')
        print('  [5-10]        will start the app 5 times with arguments 5, 6, 7, 8, 9, 10 respectively')
        print('  [5-10:1]      is equivalent to [5-10]')
        print('  [5-10:+1]     is equivalent to [5-10]')
        print('  [3-1]         will start the app 3 times with arguments 3, 2, 1 respectively')
        print('  [3-1:1]       is equivalent to [3-1]')
        print('  [3-1:-1]      is equivalent to [3-1]')
        print('  [1-16:*2]     will start the app 5 times with arguments 1, 2, 4, 8, 16 respectively')
        print('  [16-1:/2]     will start the app 5 times with arguments 16, 8, 4, 2, 1 respectively')
        print('  [foo,bar,baz] will start the app 3 times with arguments foo, bar, baz respectively')
        print()
        print('Argument groups:')
        print('  Argument groups allow to pass the same range multiple times.')
        print('  [0=1-3]       creates argument group 0 with range 1, 2, 3')
        print('  [0=]          references argument group 0, i.e. the range 1, 2, 3')
        print()
        print('Examples:')
        print('  Create sbatch file for 30 runs of ./app, with argument x = 1, ..., 10, y = 3, 4, 5 and z = 1')
        print('  {} ./app --x [1-10] -y [3-5] -z 1'.format(sys.argv[0]))
        print('  Create sbatch file for 5 runs of ./app, with argument x = y = 1, 2, 4, 8, 16, z = linear index')
        print('  {} ./app --x [0=1-16:*2] --y [0=] -z [id]'.format(sys.argv[0]))
        print('  Create sbatch file for 6 runs of ./app, with argument a = foo, bar, b = 2, 1, 0')
        print('  {} ./app -a [foo,bar] -b [2-0]'.format(sys.argv[0]))
        sys.exit()

    args = ' '.join(sys.argv[1:])

    grps = dict()
    vs = []
    def subf(match):
        if match.group(2).lower() == 'id':
            return '$SLURM_ARRAY_TASK_ID'
        if match.group(1) is not None:
            g = int(match.group(1)[:-1])
            if match.group(2):
                idx = len(vs)
                grps[g] = idx
                vs.append(match.group(2))
            else:
                idx = grps[g]
        else:
            idx = len(vs)
            vs.append(match.group(2))
        return '${{v{}}}'.format(idx)

    p = re.compile(r'\[(\d+=)?([^\]]*)\]')
    nargs = p.sub(subf, args)

    rs = [var_range(v) for v in vs]
    tot_jobs = 1
    for r in rs:
        tot_jobs *= len(r)

    for v, r in zip(vs, rs):
        strr = [str(x) for x in r]
        if len(r) > 20:
            print('{} -> {}, ..., {}'.format(v, ', '.join(strr[:10]), ', '.join(strr[-10:])), file=sys.stderr)
        else:
            print('{} -> {}'.format(v, ', '.join(strr)), file=sys.stderr)

    h = hashlib.md5()
    h.update(args)
    slurm_output = h.hexdigest()

    print('#!/bin/bash -l')
    print('#SBATCH --time=01:00:00')
    print('#SBATCH --constraint=flat,quad')
    print('#SBATCH --export=KMP_AFFINITY=balanced')
    print('#SBATCH --array=0-{}%50'.format(tot_jobs - 1))
    print('#SBATCH --output={}_%a.out'.format(slurm_output)) 
    print()

    print('# ' + args)
    print()

    for i, r in enumerate(rs):
        print('varray{}=({})'.format(i, ' '.join(str(x) for x in r)))
    print()

    print('r={}'.format('$SLURM_ARRAY_TASK_ID'))
    for i, r in enumerate(rs):
        print('d=$(($r/{}))'.format(len(r)))
        print('i{}=$(($r - $d*{}))'.format(i, len(r)))
        print('r=$d')
        print('v{0}=${{varray{0}[${{i{0}}}]}}'.format(i))
        print()

    print('if [ ! -s "{0}_${{SLURM_ARRAY_TASK_ID}}.out" ] || [ -n "$(grep -l \'srun: error\' "{0}_${{SLURM_ARRAY_TASK_ID}}.out")" ]'.format(slurm_output))
    print('then')
    print('    srun ' + nargs)
    print('fi')






