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
size = [32,64,128,256,512,1024]
bsx = [2,8,16,32,64,128,256,512,1024]
bsy = [2,8,16,32,64,128,256,512,1024]

def main():
    d = "/users/stefanm/tmp/res"
    combs = list(itertools.product(bsx, bsy))
    for s in size:    
        print("#####################")
        print(s)
        val_arr_f = []
        val_arr_d = []
        for bs_x in bsx:
            tmp_arr_f = []
            tmp_arr_d = []
            for bs_y in bsy:
                files = glob.glob(d+"/*bsx"+str(bs_x)+"_*bsy"+str(bs_y)+"_*"+str(s)+".json")
                max_vals_f = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                max_vals_d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for f in files:
                    #load file
                    data = json.load(open(f))
                    floatVals = data[0]["float"]["stencils"]
                    doubleVals = data[0]["double"]["stencils"]        
                    #get value
                    floatData = [ 
                        floatVals["copy"]["bw"], floatVals["copyi1"]["bw"],  
                        floatVals["sumi1"]["bw"], floatVals["sumj1"]["bw"], floatVals["sumk1"]["bw"],
                        floatVals["avgi"]["bw"], floatVals["avgj"]["bw"], floatVals["avgk"]["bw"],
                        floatVals["lap"]["bw"]
                    ]
                    doubleData = [ 
                        doubleVals["copy"]["bw"], doubleVals["copyi1"]["bw"],  
                        doubleVals["sumi1"]["bw"], doubleVals["sumj1"]["bw"], doubleVals["sumk1"]["bw"],
                        doubleVals["avgi"]["bw"], doubleVals["avgj"]["bw"], doubleVals["avgk"]["bw"],
                        doubleVals["lap"]["bw"]
                    ]
                    if max_vals_f[0] < floatData[0]:
                        max_vals_f[0] = floatData[0]
                    if max_vals_f[1] < floatData[1]:
                        max_vals_f[1] = floatData[1]

                    if max_vals_f[2] < floatData[2]:
                        max_vals_f[2] = floatData[2] 
                    if max_vals_f[3] < floatData[3]:
                        max_vals_f[3] = floatData[3] 
                    if max_vals_f[4] < floatData[4]:
                        max_vals_f[4] = floatData[4] 

                    if max_vals_f[5] < floatData[5]:
                        max_vals_f[5] = floatData[5] 
                    if max_vals_f[6] < floatData[6]:
                        max_vals_f[6] = floatData[6] 
                    if max_vals_f[7] < floatData[7]:
                        max_vals_f[7] = floatData[7] 

                    if max_vals_f[8] < floatData[8]:
                        max_vals_f[8] = floatData[8] 

                    if max_vals_d[0] < doubleData[0]:
                        max_vals_d[0] = doubleData[0] 
                    if max_vals_d[1] < doubleData[1]:
                        max_vals_d[1] = doubleData[1] 

                    if max_vals_d[2] < doubleData[2]:
                        max_vals_d[2] = doubleData[2] 
                    if max_vals_d[3] < doubleData[3]:
                        max_vals_d[3] = doubleData[3] 
                    if max_vals_d[4] < doubleData[4]:
                        max_vals_d[4] = doubleData[4] 

                    if max_vals_d[5] < doubleData[5]:
                        max_vals_d[5] = doubleData[5] 
                    if max_vals_d[6] < doubleData[6]:
                        max_vals_d[6] = doubleData[6] 
                    if max_vals_d[7] < doubleData[7]:
                        max_vals_d[7] = doubleData[7] 

                    if max_vals_d[8] < doubleData[8]:
                        max_vals_d[8] = doubleData[8]
                #print "##################################"
                #print d  
                #print bs_x
                #print bs_y
                #print s  
#
                #print max_vals_f[0]
                #print max_vals_f[1]
                #print max_vals_f[2]
                #print max_vals_f[3]
                #print max_vals_f[4]
                #print max_vals_f[5]
                #print max_vals_f[6]
                #print max_vals_f[7]
                #print max_vals_f[8]
                #print "     "
                #print max_vals_d[0]
                #print max_vals_d[1]
                #print max_vals_d[2]
                #print max_vals_d[3]
                #print max_vals_d[4]
                #print max_vals_d[5]
                #print max_vals_d[6]
                #print max_vals_d[7]
                #print max_vals_d[8]
                # append all the bsy information to arr
                tmp_arr_f.append(round(np.mean(max_vals_f), 3))
                tmp_arr_d.append(round(np.mean(max_vals_d), 3))
            # append arr to vals            
            val_arr_f.append(tmp_arr_f)
            val_arr_d.append(tmp_arr_d)
        # create plot out of vals
        plt.figure(figsize=(15, 15))
        plt.suptitle("i-first layout, double prec., domain size "+str(s), fontsize=25)
        x = np.array([0,1,2,3,4,5,6,7,8])
        y = np.array([0,1,2,3,4,5,6,7,8])
        plt.xticks(x, ["2","8","16","32","64","128","256","512","1024"])
        plt.yticks(y, ["2","8","16","32","64","128","256","512","1024"])
        plt.xlabel('Blocksize J')
        plt.ylabel('Blocksize I')
        maxed = max([max(sub_array) for sub_array in val_arr_d])
        print(maxed)
        ix = plt.imshow(val_arr_d, cmap=plt.get_cmap("gist_rainbow"), interpolation='bicubic', vmin=0, vmax=maxed )
        cb = plt.colorbar(ix)
        cb.set_label('GB/s')
        plt.savefig("hm_double_"+str(s)+".png")

        plt.figure(figsize=(15, 15))
        plt.suptitle("i-first layout, float prec., domain size "+str(s), fontsize=25)
        x = np.array([0,1,2,3,4,5,6,7,8])
        y = np.array([0,1,2,3,4,5,6,7,8])
        plt.xticks(x, ["2","8","16","32","64","128","256","512","1024"])
        plt.yticks(y, ["2","8","16","32","64","128","256","512","1024"])
        plt.xlabel('Blocksize J')
        plt.ylabel('Blocksize I')
        maxed = max([max(sub_array) for sub_array in val_arr_f])
        print(maxed)
        ix = plt.imshow(val_arr_f, cmap=plt.get_cmap("gist_rainbow"), interpolation='bicubic', vmin=0, vmax=maxed )
        cb = plt.colorbar(ix)
        cb.set_label('GB/s')
        plt.savefig("hm_float_"+str(s)+".png")

if __name__ == "__main__":
    main()

