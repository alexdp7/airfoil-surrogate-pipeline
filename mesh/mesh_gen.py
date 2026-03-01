import numpy as np
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--coords', type=str, required=True, help='Path to airfoil CSV')
coords = np.loadtxt('test_2412_5.csv', delimiter=',', skiprows=1)

x, y = coords[:,0],coords[:,1]
R = 20 #taking 1 as reference upstream  approx 20 chords
middle = len(x)//2
#vertices location

P0 = [x[middle],y[middle]]
P1 = [x[0],y[0]]
P2 = [x[-1], y[-1]]

P3 = [0.25,R]
P4 = [-R+0.25,0]
P5 = [0.25,-R]

P6 = [1.0, (y[0] + y[-1]) / 2.0] 
P7  = [1.0, R]    
P8  = [1.0, -R]    
P9  = [30, R]      
P10 = [30, 0.0]   
P11 = [30, -R]     
print(x[0], y[0])   # should be TE upper after rotation
print(x[-1], y[-1]) # should be TE lower after rotation
print(x[middle], y[middle])  # should be LE2)

def write_vertices(vertices):
    lines = ["vertices\n(\n"]
    for p in vertices:
        lines.append(f"    ({p[0]} {p[1]} 0)\n")
    for p in vertices:
        lines.append(f"    ({p[0]} {p[1]} 0.1)\n")
    lines.append(");\n")
    return "".join(lines)

vertices = [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11]
print(write_vertices(vertices))