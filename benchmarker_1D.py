import itertools
import subprocess
import sys, getopt
import os

alignments = [1,2,4,8,16,32]
threads = [2,8,16,32,64,128,256]
layout = ["2,1,0", "2,0,1", "1,0,2", "1,2,0", "0,1,2", "0,2,1"]
execution = ["benchmark_1D"]
mode = ["CACHE_MODE", "FLAT_MODE"]
isi = 0
jsi = 0
ksi = 0
dire = ""

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print proc_stdout

def compile(ex, mo, al, la):
    # compile with execution, mode, blocksize, alignment, layout
    output_name = ex.lower()+"_"+mo.lower()+"_al"+str(al)+"_la"+la.replace(",","-")
    cmd = "CC ./"+ex+"/main.cpp --std=c++11 -DNDEBUG -D"+mo+" -DLAYOUT="+la+" -DALIGN="+str(al)+" -I/users/stefanm/boost_1_62_0/include -I./gridtools_storage/include -I./libjson -DJSON_ISO_STRICT -L./libjson -ljson  -lmemkind -O3 -fopenmp -o "+dire+"/"+output_name
    subprocess_cmd(cmd)
    #print cmd
    print "compiled: "+output_name

def execute(ex, mo, al, th, la):
    output_name = ex.lower()+"_"+mo.lower()+"_al"+str(al)+"_la"+la.replace(",","-")
    if mo == "CACHE_MODE":
        subprocess_cmd("OMP_NUM_THREADS="+str(th)+" salloc --exclusive -C cache,quad srun ./"+output_name+" --isize "+str(isi)+" --jsize "+str(jsi)+" --ksize "+str(ksi)+" --write")
        #print "OMP_NUM_THREADS="+str(th)+" salloc --exclusive -C cache,quad srun ./"+output_name+" --isize "+str(isi)+" --jsize "+str(jsi)+" --ksize "+str(ksi)+" --write"
    elif mo == "FLAT_MODE":
        subprocess_cmd("OMP_NUM_THREADS="+str(th)+" salloc --exclusive -C flat,quad srun numactl --membind=1 ./"+output_name+" --isize "+str(isi)+" --jsize "+str(jsi)+" --ksize "+str(ksi)+" --write")
        #print "OMP_NUM_THREADS="+str(th)+" salloc --exclusive -C flat,quad srun numactl --membind=1 ./"+output_name+" --isize "+str(isi)+" --jsize "+str(jsi)+" --ksize "+str(ksi)+" --write"
    else:
        sys.exit("mode unknown")
        

def main(argv):
    global isi, jsi, ksi, dire
    try:
        opts, args = getopt.getopt(argv,"hi:j:k:",["isize=","jsize=","ksize="])
    except getopt.GetoptError:
        print 'benchmarker.py -i <isize> -j <jsize> -k <ksize>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'benchmarker.py -i <isize> -j <jsize> -k <ksize>'
            sys.exit()
        elif opt in ("-i", "--isize"):
            isi = arg
        elif opt in ("-j", "--jsize"):
            jsi = arg
        elif opt in ("-k", "--ksize"):
            ksi = arg
    if isi==0 or jsi==0 or ksi==0:
        print 'benchmarker.py -i <isize> -j <jsize> -k <ksize>'
        sys.exit()        
    dire = "./test/benchmark_1D_" + str(isi) + "x" + str(jsi) + "x" + str(ksi)
    subprocess_cmd("source ./setup.sh")
    subprocess_cmd("mkdir "+dire)
    subprocess_cmd("pwd")

    combs1 = list(itertools.product(execution, mode, alignments, layout))
    i = len(combs1)-1
    for cc in combs1:
        # extract data
        ex = cc[0]
        mo = cc[1]
        al = cc[2]
        la = cc[3]
        compile(ex, mo, al, la)
        print str(i) + " left"
        i = i-1
    
    print "COMPILATION DONE"

    os.chdir(dire)
    subprocess_cmd("pwd")
    combs = list(itertools.product(execution, mode, alignments, threads, layout))
    i = len(combs)-1
    for cc in combs:
        # extract data
        ex = cc[0]
        mo = cc[1]
        al = cc[2]
        th = cc[3]
        la = cc[4]
        #compile(ex, mo, bsx, bsy, al)
        execute(ex, mo, al, th, la)
        print str(i) + " left"
        i = i-1

if __name__ == "__main__":
    main(sys.argv[1:])
