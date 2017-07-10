import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import glob
import itertools
import subprocess
import sys
import json
from pprint import pprint
import os.path

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()

alignments = [1,32]
threads = [32,64,128,256]
execution = ["benchmark_ij_parallel"]
mode = ["CACHE_MODE", "FLAT_MODE"]
block_size_x = [1,2,8,16,32,64,128,256,512,1024]
block_size_y = [1,2,8,16,32,64,128,256,512,1024]
size = ["64x64x80", "128x128x80", "256x256x80", "512x512x80", "1024x1024x80"]
layout = ["2,1,0"] #, "2,0,1", "1,0,2", "1,2,0", "0,1,2", "0,2,1"]
prec = ["float", "double"]

dire = "./test/benchmark_ij_ifirst_plots"
subprocess_cmd("mkdir "+dire)

# create combinatorial of mode, alignment, layout
combinations = list(itertools.product(mode, alignments, layout, prec, block_size_x, block_size_y))
for e in combinations:
    m = e[0].lower()
    a = e[1]
    l = e[2].replace(",","-")
    p = e[3]
    bsx = e[4]
    bsy = e[5]
    outname = dire+"/"+p+"_m"+m+"_bsx"+str(bsx)+"_bsy"+str(bsy)+"_a"+str(a)+".svg"
    if os.path.isfile(outname):
        print("already exists " + outname)
        continue  
    if int(bsx) < int(8) and int(bsy) < int(8):
        continue

    #create plot figure
    plt.figure(figsize=(30, 15))
    plt.suptitle(str(execution) + " BSX " + str(bsx) + " BSY " + str(bsy) + " Alignment " + str(a) + " Mode " + str(m) + " " + p, fontsize=25)
    x = np.array([0,1,2,3,4])
    # iterate through threads
    gs = gridspec.GridSpec(2, 4)     
    i = 0
    for t in threads:
        #iterate through domain sizes
        copy = []
        copyi1 = []
        sumi1 = []
        sumj1 = []
        sumk1 = []
        avgi = []
        avgj = []
        avgk = []
        lap = []

        for s in size:
            # get data for given size, and number of threads, and config
            filename = "./test/benchmark_ij_"+s+"/res_benchmark_ij_m"+m+"_a"+str(a)+"_l"+l+"_t"+str(t)+"_bsx"+str(bsx)+"_bsy"+str(bsy)+".json"
            if os.path.isfile(filename):
                json_data = open(filename, "r")
                data = json.load(json_data)
                all_stencils = data[0][p]["stencils"]
                copy.append(all_stencils["copy"]["bw"])
                copyi1.append(all_stencils["copyi1"]["bw"])
                sumi1.append(all_stencils["sumi1"]["bw"])
                sumj1.append(all_stencils["sumj1"]["bw"])
                sumk1.append(all_stencils["sumk1"]["bw"])
                avgi.append(all_stencils["avgi"]["bw"])
                avgj.append(all_stencils["avgj"]["bw"])
                avgk.append(all_stencils["avgk"]["bw"])
                lap.append(all_stencils["lap"]["bw"])
                json_data.close()
            else:
                copy.append(0)
                copyi1.append(0)
                sumi1.append(0)
                sumj1.append(0)
                sumk1.append(0)
                avgi.append(0)
                avgj.append(0)
                avgk.append(0)
                lap.append(0)


        plt.subplot(gs[i])
        i = i+1
        plt.ylim([0,500])
        plt.title('Threads '+str(t))
        plt.xticks(x, ["64","128","256","512","1024"])
        plt.plot(x, copy,'r-', label="copy", linewidth=2)
        plt.plot(x, copyi1,'g-', label="copyi1", linewidth=2)
        plt.plot(x, sumi1,'r-.', label="sumi1", linewidth=2)
        plt.plot(x, sumj1,'g-.', label="sumj1", linewidth=2)
        plt.plot(x, sumk1,'b-.', label="sumk1", linewidth=2)
        plt.plot(x, avgi,'r--', label="avgi", linewidth=2)
        plt.plot(x, avgj,'g--', label="avgj", linewidth=2)
        plt.plot(x, avgk,'b--', label="avgk", linewidth=2)
        plt.plot(x, lap,'k-', label="lap", linewidth=2)
        plt.ylabel('GB/s')
        plt.xlabel('Domain size')
        plt.grid(True)

    plt.subplot(gs[i])
    plt.title('Legend')
    plt.plot([], [],'r-', label="copy")
    plt.plot([], [],'g-', label="copyi1")
    plt.plot([], [],'r-.', label="sumi1")
    plt.plot([], [],'g-.', label="sumj1")
    plt.plot([], [],'b-.', label="sumk1")
    plt.plot([], [],'r--', label="avgi")
    plt.plot([], [],'g--', label="avgj")
    plt.plot([], [],'b--', label="avgk")
    plt.plot([], [],'k-', label="lap")
    plt.legend()

    plt.savefig(outname)
    plt.close()
    print("wrote " + outname)
