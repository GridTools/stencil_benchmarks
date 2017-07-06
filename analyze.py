import glob
import itertools
import subprocess
import sys
import json
from pprint import pprint
import numpy as np

def main():
    dirs = glob.glob("./test/benchmark_ij_10*")
    for d in dirs:
        files = glob.glob(d+"/*.json")
        max_vals_f = [ 
            ["copy", "", 0.0], ["copyi1", "", 0.0], 
            ["sumi1", "", 0.0], ["sumj1", "", 0.0], ["sumk1", "", 0.0], 
            ["avgi", "", 0.0], ["avgj", "", 0.0], ["avgk", "", 0.0], 
            ["lap", "", 0.0]
        ]
        max_vals_d = [ 
            ["copy", "", 0.0], ["copyi1", "", 0.0], 
            ["sumi1", "", 0.0], ["sumj1", "", 0.0], ["sumk1", "", 0.0], 
            ["avgi", "", 0.0], ["avgj", "", 0.0], ["avgk", "", 0.0], 
            ["lap", "", 0.0]
        ]
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
            if max_vals_f[0][2] < floatData[0]:
                max_vals_f[0] = [ "copy", f, floatData[0] ]
            if max_vals_f[1][2] < floatData[1]:
                max_vals_f[1] = [ "copyi1", f, floatData[1] ]

            if max_vals_f[2][2] < floatData[2]:
                max_vals_f[2] = [ "sumi1", f, floatData[2] ]
            if max_vals_f[3][2] < floatData[3]:
                max_vals_f[3] = [ "sumj1", f, floatData[3] ]
            if max_vals_f[4][2] < floatData[4]:
                max_vals_f[4] = [ "sumk1", f, floatData[4] ]

            if max_vals_f[5][2] < floatData[5]:
                max_vals_f[5] = [ "avgi", f, floatData[5] ]
            if max_vals_f[6][2] < floatData[6]:
                max_vals_f[6] = [ "avgj", f, floatData[6] ]
            if max_vals_f[7][2] < floatData[7]:
                max_vals_f[7] = [ "avgk", f, floatData[7] ]
                
            if max_vals_f[8][2] < floatData[8]:
                max_vals_f[8] = [ "lap", f, floatData[8] ]

            if max_vals_d[0][2] < doubleData[0]:
                max_vals_d[0] = [ "copy", f, doubleData[0] ]
            if max_vals_d[1][2] < doubleData[1]:
                max_vals_d[1] = [ "copyi1", f, doubleData[1] ]

            if max_vals_d[2][2] < doubleData[2]:
                max_vals_d[2] = [ "sumi1", f, doubleData[2] ]
            if max_vals_d[3][2] < doubleData[3]:
                max_vals_d[3] = [ "sumj1", f, doubleData[3] ]
            if max_vals_d[4][2] < doubleData[4]:
                max_vals_d[4] = [ "sumk1", f, doubleData[4] ]

            if max_vals_d[5][2] < doubleData[5]:
                max_vals_d[5] = [ "avgi", f, doubleData[5] ]
            if max_vals_d[6][2] < doubleData[6]:
                max_vals_d[6] = [ "avgj", f, doubleData[6] ]
            if max_vals_d[7][2] < doubleData[7]:
                max_vals_d[7] = [ "avgk", f, doubleData[7] ]
                
            if max_vals_d[8][2] < doubleData[8]:
                max_vals_d[8] = [ "lap", f, doubleData[8] ]  
        print "##################################"
        print d          
        print max_vals_f[0]
        print max_vals_f[1]
        print max_vals_f[2]
        print max_vals_f[3]
        print max_vals_f[4]
        print max_vals_f[5]
        print max_vals_f[6]
        print max_vals_f[7]
        print max_vals_f[8]
        print "     "
        print max_vals_d[0]
        print max_vals_d[1]
        print max_vals_d[2]
        print max_vals_d[3]
        print max_vals_d[4]
        print max_vals_d[5]
        print max_vals_d[6]
        print max_vals_d[7]
        print max_vals_d[8]

    
if __name__ == "__main__":
    main()
