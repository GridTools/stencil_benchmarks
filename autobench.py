#!/usr/bin/env python

import click
import collections
import itertools
import json
import os
import shutil
import subprocess
import time
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from benchvars import *


def suffix(options):
    so = sorted(options, key=lambda o: o.name)
    return '_'.join(o.short_str() for o in so)


def binary_name(options):
    return 'stencil_bench_' + suffix(options)


def result_name(options):
    return 'res_benchmark_' + suffix(options) + '.json'


def slurmout_name(options):
    return 'slurm_' + suffix(options) + '.out'


def sbatch_name(options):
    return 'sbatch_' + suffix(options) + '.sh'


def plot_name(options, prec):
    return 'plot_' + suffix(options) + '_' + prec + '.svg'


def compile_binaries(vos, outdir):
    for options in vos.generate(VarType.compiletime):
        outname = os.path.join(outdir, binary_name(options))
        print('Compiling {}'.format(outname))
        with open('Makefile.config', 'w') as mf:
            mf.write('\n'.join(str(o) for o in options))
        subprocess.call(['make', 'clean'], stdout=subprocess.PIPE)
        try:
            subprocess.check_call(['make'], stdout=subprocess.PIPE)
        except subprocess.CalledProcessError:
            print('FATAL ERROR: compilation failed')
            sys.exit(1)
        shutil.move('stencil_bench', outname)


def create_sbatch(vos, outdir):
    for ctoptions in vos.generate(VarType.compiletime):
        bname = binary_name(ctoptions)
        print('Generating sbatch files for binary {}'.format(bname))
        mcdram = next(o.value.lower() for o in ctoptions if o.name == 'MCDRAM')
        for rtoptions in vos.generate(VarType.runtime):
            for envoptions in vos.generate(VarType.environment):
                options = ctoptions + rtoptions + envoptions
                outname = os.path.join(outdir, sbatch_name(options))

                envstr = '\n'.join('#SBATCH --export={}'.format(str(o)) for o in envoptions)
                rtstr = ' '.join(str(o) for o in rtoptions) + ' --write ' + result_name(options)

                envstr += '\n#SBATCH --export=KMP_AFFINITY=balanced'

                with open(outname, 'w') as bf:
                    bf.write('#!/bin/bash -l\n')
                    bf.write('#SBATCH --job-name={}\n'.format(suffix(options)))
                    bf.write('#SBATCH --time=01:00:00\n')
                    bf.write('#SBATCH --constraint={},quad\n'.format(mcdram))
                    bf.write('#SBATCH --output={}\n'.format(slurmout_name(options)))
                    bf.write(envstr)
                    bf.write('\nsrun ./{} {}\n'.format(bname, rtstr))


def commit_sbatch(vos, outdir, wait, maxrun, failed_only):
    jobids = []
    for options in vos.generate():
        sbname = sbatch_name(options)
        sbpath = os.path.join(outdir, sbname)
        if failed_only and os.path.isfile(os.path.join(outdir, result_name(options))):
            continue

        if os.path.isfile(sbpath):
            print('Commiting {}'.format(sbname))
            while True:
                try:
                    sbatch_output = subprocess.check_output(['sbatch', sbname], cwd=outdir)
                    break
                except subprocess.CalledProcessError:
                    print('Submitting job failed, retry...')
                    time.sleep(1)
            jobids.append(sbatch_output.split()[-1])
            time.sleep(1)

            if maxrun > 0:
                waittime = 10
                while True:
                    nrunning = subprocess.check_output(['squeue']).count('\n') - 1
                    if nrunning < maxrun:
                        break
                    time.sleep(waittime)
                    waittime = min(2 * waittime, 300)

    if wait and jobids:
        print('Waiting for jobs to finish...')
        with open(os.path.join(outdir, 'wait.sh'), 'w') as wf:
            wf.write('#!/bin/bash -l\n')
            wf.write('#SBATCH --dependency=afterany:{}\n'.format(':'.join(jobids)))
            wf.write('#SBATCH --time=00:01:00\n')
            wf.write('#SBATCH --output={}\n'.format('wait.out'))
            wf.write('#SBATCH --wait\n')
            wf.write('srun echo "finished!"\n')
        while True:
            try:
                subprocess.check_call(['sbatch', 'wait.sh'], cwd=outdir, stdout=subprocess.PIPE)
                break
            except subprocess.CalledProcessError:
                    print('Submitting job failed, retry...')
                    time.sleep(1)


def plot_results(vos, outdir):
    matplotlib.rcParams.update({'font.size': 25, 'xtick.labelsize': 20})
    for ctoptions in vos.generate(VarType.compiletime):
        print('Create plot for compile-time options {}'.format(' '.join(str(o) for o in ctoptions)))
        
        float_data = collections.defaultdict(lambda: collections.defaultdict(list))
        double_data = collections.defaultdict(lambda: collections.defaultdict(list))
        for rtoptions in vos.generate(VarType.runtime):
            for envoptions in vos.generate(VarType.environment):
                options = ctoptions + rtoptions + envoptions
                rname = os.path.join(outdir, result_name(options))
                try:
                    with open(rname, 'r') as f:
                        try:
                            jf = json.load(f)[0]
                        except ValueError:
                            print('Could not load JSON data from file {}!'.format(rname))
                            continue
                        size = jf['x']
                        threads = jf['threads']
                        float_stencils = jf['float']['stencils']
                        for stencil, results in float_stencils.items():
                            float_data[threads][stencil].append((size, results['bw']))
                        double_stencils = jf['double']['stencils']
                        for stencil, results in double_stencils.items():
                            double_data[threads][stencil].append((size, results['bw']))
                except IOError:
                    print('Expected file {} not found!'.format(rname))

        for prec, data in [('single', float_data), ('double', double_data)]:
            fig, axs = plt.subplots(2, 2, figsize=(30, 20))
            fig.suptitle(' - '.join(str(o) for o in ctoptions) + ' - ' + prec.upper() + ' PREC.')

            for ax, threads in zip(axs.flat, sorted(float_data.keys())):
                for stencil in sorted(float_data[threads].keys()):
                    ax.set_title('{} Threads'.format(threads))
                    x, y = zip(*data[threads][stencil])
                    ax.semilogx(x, y, basex=2, lw=2, ls='--', label=stencil)
                    ax.set_xlim([0, 1024])
                    ax.set_ylim([0, 500])
                    ax.set_xlabel('Domain Size')
                    ax.set_ylabel('Estimated Bandwidth [GB/s]')
                    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda s, _: '{0}x{0}x80'.format(int(s))))
                    ax.grid(True)
                ax.legend(ncol=3, loc='upper left')

            fig.savefig(os.path.join(outdir, plot_name(ctoptions, prec[0])))
            plt.close(fig)


vos = None
outdir = None

@click.group(chain=True)
@click.option('--stencil', '-s', multiple=True, default=['1D', 'ij_parallel_i_first', 'ij_parallel_k_first', 'k_parallel'])
@click.option('--align', '-a', multiple=True, default=[1, 32], type=int)
@click.option('--layout', '-l', multiple=True, default=list(itertools.permutations((0, 1, 2))), type=(int, int, int))
@click.option('--mcdram', '-r', multiple=True, default=['FLAT', 'CACHE'])
@click.option('--blocksize', '-b', multiple=True, default=[1, 16, 32, 64], type=int)
@click.option('--threads', '-t', multiple=True, default=[32, 64, 128, 256], type=int)
@click.option('--size', '-x', multiple=True, default=[32, 64, 128, 256, 512, 1024], type=int)
@click.argument('directory', type=click.Path(exists=True))
def cli(stencil, align, layout, mcdram, blocksize, threads, size, directory):
    global vos
    global outdir
    
    vos = VarOptionsList()
    vos.add_ct('STENCIL', stencil)
    vos.add_ct('ALIGN', align)
    vos.add_ct('LAYOUT', layout, 
               fmt=lambda v: 'LAYOUT={},{},{}'.format(*v),
               sfmt=lambda v: 'l{}{}{}'.format(*v))
    vos.add_ct('MCDRAM', [m.upper() for m in mcdram])
    blocksize = [(x, y) for x, y in itertools.product(blocksize, blocksize) if x >= 8 or y >= 8]
    if blocksize:
        vos.add_ct('BLOCKSIZE', blocksize,
                   fmt=lambda v: 'BLOCKSIZEX={}\nBLOCKSIZEY={}'.format(*v),
                   sfmt=lambda v: 'bsx{}_bsy{}'.format(*v))
    vos.add_env('OMP_NUM_THREADS', threads, sfmt=lambda v: 't{}'.format(v))
    vos.add_rt('size', size, fmt=lambda v: '--isize {0} --jsize {0}'.format(v))

    outdir = directory

    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')


@cli.command()
def report():
    print(vos)
    print('Working directory: {}'.format(outdir))


@cli.command()
def compile():
    compile_binaries(vos, outdir)


@cli.command()
def sbatch():
    create_sbatch(vos, outdir)


@cli.command()
@click.option('--wait/--no-wait', '-w', default=False)
@click.option('--maxrun', '-m', default=100)
@click.option('--failed-only', is_flag=True, default=False)
def commit(wait, maxrun, failed_only):
    commit_sbatch(vos, outdir, wait, maxrun, failed_only)


@cli.command()
def plot():
    plot_results(vos, outdir)


@cli.command()
@click.option('--maxrun', '-m', default=100)
def all(maxrun):
    print(vos)
    print('Working directory: {}'.format(outdir))
    compile_binaries(vos, outdir)
    create_sbatch(vos, outdir)
    commit_sbatch(vos, outdir, True, maxrun, False)
    plot_results(vos, outdir)


if __name__ == '__main__':
    cli()
