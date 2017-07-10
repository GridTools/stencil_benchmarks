import itertools
import subprocess
import sys, getopt
import os
import time

alignments = [1,32]
threads = [32,64,128,256]
execution = ["benchmark_ij_parallel"]
mode = ["CACHE_MODE", "FLAT_MODE"]
block_size_x = [1,2,8,16,32,64,128,256,512,1024]
block_size_y = [1,2,8,16,32,64,128,256,512,1024]
isi = 0
jsi = 0
ksi = 0
dire = ""
count = 0

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print proc_stdout

def compile(ex, mo, bsx, bsy, al):
    if int(bsx) >= int(8) or int(bsy) >= int(8):
        if int(bsx) <= int(isi) and int(bsy) <= int(jsi):
            # compile with execution, mode, blocksize, alignment, layout
            output_name = ex.lower()+"_"+mo.lower()+"_bsx"+str(bsx)+"_bsy"+str(bsy)+"_al"+str(al) #"_la"+la.replace(",","-")
            cmd = "CC ./"+ex+"/main.cpp --std=c++11 -DNDEBUG -D"+mo+" -DBLOCKSIZEX="+str(bsx)+" -DBLOCKSIZEY="+str(bsy)+" -DALIGN="+str(al)+" -I/users/stefanm/boost_1_62_0/include -I./gridtools_storage/include -I./libjson -DJSON_ISO_STRICT -L./libjson -ljson  -lmemkind -O3 -fopenmp -o "+dire+"/"+output_name
            subprocess_cmd(cmd)
            print "compiled: "+output_name
        else:
            print "compilation skipped bsx " + str(bsx) + " bsy " + str(bsy) + " isize " + str(isi) + " jsize " + str(jsi)
    else:
        print "compilation skipped bsx " + str(bsx) + " bsy " + str(bsy)


def execute(ex, mo, bsx, bsy, al, th):
    global count
    outname = "res_benchmark_ij_m"+mo.lower()+"_a"+str(al)+"_l2-1-0_t"+str(th)+"_bsx"+str(bsx)+"_bsy"+str(bsy)+".json"
    if os.path.isfile(outname):
        print("already exists " + outname)
    else:
        if int(bsx) >= int(8) or int(bsy) >= int(8):
            output_name = ex.lower()+"_"+mo.lower()+"_bsx"+str(bsx)+"_bsy"+str(bsy)+"_al"+str(al) #"_la"+la.replace(",","-")
            if int(bsx) <= int(isi) and int(bsy) <= int(jsi):
                if mo == "CACHE_MODE":
                    subprocess_cmd("OMP_NUM_THREADS="+str(th)+" salloc --exclusive -C cache,quad srun ./"+output_name+" --isize "+str(isi)+" --jsize "+str(jsi)+" --ksize "+str(ksi)+" --write &> /dev/null &")
                    print "OMP_NUM_THREADS="+str(th)+" salloc --exclusive -C cache,quad srun ./"+output_name+" --isize "+str(isi)+" --jsize "+str(jsi)+" --ksize "+str(ksi)+" --write"
                    count = count+1
                elif mo == "FLAT_MODE":
                    subprocess_cmd("OMP_NUM_THREADS="+str(th)+" salloc --exclusive -C flat,quad srun numactl --membind=1 ./"+output_name+" --isize "+str(isi)+" --jsize "+str(jsi)+" --ksize "+str(ksi)+" --write &> /dev/null &")
                    print "OMP_NUM_THREADS="+str(th)+" salloc --exclusive -C flat,quad srun numactl --membind=1 ./"+output_name+" --isize "+str(isi)+" --jsize "+str(jsi)+" --ksize "+str(ksi)+" --write"
                    count = count+1
                else:
                    sys.exit("mode unknown")
    if count>150:
        print "sleep five minutes"
        time.sleep(250) 
        count = 0

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
    dire = "./test/benchmark_ij_" + str(isi) + "x" + str(jsi) + "x" + str(ksi)
    subprocess_cmd("source ./setup.sh")
    subprocess_cmd("mkdir "+dire)
    subprocess_cmd("pwd")

    combs1 = list(itertools.product(execution, mode, block_size_x, block_size_y, alignments))
    i = len(combs1)-1
    for cc in combs1:
        # extract data
        ex = cc[0]
        mo = cc[1]
        bsx = cc[2]
        bsy = cc[3]
        al = cc[4]
        #compile(ex, mo, bsx, bsy, al)
        print str(i) + " left"
        i = i-1
    
    print "COMPILATION DONE"

    os.chdir(dire)
    subprocess_cmd("pwd")
    combs = list(itertools.product(execution, mode, block_size_x, block_size_y, alignments, threads))
    i = len(combs)-1
    for cc in combs:
        # extract data
        ex = cc[0]
        mo = cc[1]
        bsx = cc[2]
        bsy = cc[3]
        al = cc[4]
        th = cc[5]
        #compile(ex, mo, bsx, bsy, al)
        execute(ex, mo, bsx, bsy, al, th)
        print str(i) + " left"
        i = i-1

if __name__ == "__main__":
    main(sys.argv[1:])
