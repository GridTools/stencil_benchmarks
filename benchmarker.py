import itertools
import subprocess
import sys

alignments = [4,8,16,32]
layouts = ["2,1,0", "2,0,1", "1,2,0", "1,0,2", "0,1,2", "0,2,1"]
threads = [2,8,16,32,64,128,256]
execution = ["benchmark_ij_parallel", "benchmark_k_parallel"]
mode = ["CACHE_MODE", "FLAT_MODE"]
block_size = [8,16,32]

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print proc_stdout

def compile(ex, mo, bs, al, la):
    # compile with execution, mode, blocksize, alignment, layout
    output_name = ex.lower()+"_"+mo.lower()+"_bs"+str(bs)+"_al"+str(al)+"_la"+la.replace(",","-")
    cmd = "CC ./"+ex+"/main.cpp --std=c++11 -DNDEBUG -D"+mo+" -DBLOCKSIZEX="+str(bs)+" -DBLOCKSIZEY="+str(bs)+" -DALIGN="+str(al)+" -DLAYOUT="+str(la)+" -I/users/stefanm/boost_1_62_0/include -I./gridtools_storage/include -I./benchmark_ij_parallel/libjson -DJSON_ISO_STRICT -L./benchmark_ij_parallel/libjson -ljson  -lmemkind -O3 -fopenmp -o ./test/"+output_name
    subprocess_cmd(cmd)
    print "compiled: "+output_name

def execute(ex, mo, bs, al, la, th):
    output_name = ex.lower()+"_"+mo.lower()+"_bs"+str(bs)+"_al"+str(al)+"_la"+la.replace(",","-")
    if mo == "CACHE_MODE":
        subprocess_cmd("OMP_NUM_THREADS="+str(th)+" salloc -C cache,quad srun ./"+output_name)
    elif mo == "FLAT_MODE":
        subprocess_cmd("OMP_NUM_THREADS="+str(th)+" salloc -C flat,quad srun numactl --membind=1 ./"+output_name)
    else:
        sys.exit("mode unknown")
        

def main():
    subprocess_cmd("source ../setup.sh")
    #subprocess_cmd("mkdir test")
    subprocess_cmd("pwd")

    combs = list(itertools.product(execution, mode, threads, block_size, alignments, layouts))
    i = len(combs)-1
    for cc in combs:
        # extract data
        ex = cc[0]
        mo = cc[1]
        th = cc[2]
        bs = cc[3]
        al = cc[4]
        la = cc[5]
        #compile(ex, mo, bs, al, la)
        execute(ex, mo, bs, al, la, th)
        print str(i) + " left"
        i = i-1

if __name__ == "__main__":
    main()
