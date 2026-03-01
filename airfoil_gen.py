#generate the naca airfoil points
import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_naca(airfoil): # later it needs to loop through a text file to get the names of the airfoils
    thickness = int(airfoil[2:])/100 #last 2 digits are the thickness
    m = int(airfoil[0]) /100.0
    p = int(airfoil[1])/ 10.0
    points = 256
    split = points//2
    beta = np.linspace(0,np.pi, points//2)
#start the clustering at the leading edge and trailing edge to better define the curves

    x = np.zeros_like(beta)
    x_upper = np.zeros_like(beta)
    x_lower = np.zeros_like(beta)
    y_upper = np.zeros_like(beta)
    y_lower = np.zeros_like(beta)
    x_surf = np.zeros_like(beta)
    y_surf = np.zeros_like(beta)

    def clustering(beta):
        return  0.5 * (1-np.cos(beta))
    
    x=clustering(beta)
    

    def yt(x, t = thickness):
        return 5 * t *(0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516*x**2 +0.2843*x**3 -0.1015*x**4)
    
    def yc_minus(x, p = p, m=m):
        return (m/p**2)*(2*p*x-x**2)
    

    def yc_plus(x, p = p, m=m):
        return (m/(1-p)**2)*(1-2*p+2*p*x-x**2)
    

    def theta_plus(x, p=p, m=m):
        return  np.arctan((m/(1-p)**2)*(2*p-2*x))
    
    def theta_minus(x, p=p, m=m):
        return np.arctan((m/p**2)*(2*p-2*x))
    
    if m>0:
        i1 = np.where(x<=p)
        i2 = np.where(x>p)

        
        x_upper[i1] = x[i1] - yt(x[i1]) * np.sin(theta_minus(x[i1]))
        y_upper[i1] = yc_minus(x[i1]) + yt(x[i1]) * np.cos(theta_minus(x[i1]))
        x_lower[i1] = x[i1] + yt(x[i1]) * np.sin(theta_minus(x[i1]))
        y_lower[i1] = yc_minus(x[i1]) - yt(x[i1]) * np.cos(theta_minus(x[i1]))


        x_upper[i2] = x[i2] - yt(x[i2]) * np.sin(theta_plus(x[i2]))
        y_upper[i2] = yc_plus(x[i2]) + yt(x[i2]) * np.cos(theta_plus(x[i2]))
        x_lower[i2] = x[i2] + yt(x[i2]) * np.sin(theta_plus(x[i2]))
        y_lower[i2] = yc_plus(x[i2]) - yt(x[i2]) * np.cos(theta_plus(x[i2]))
    else:
        x_upper = x
        x_lower = x
        y_upper = yt(x_upper)
        y_lower = -yt(x_lower)

    x_surf = np.concatenate([np.flip(x_upper),x_lower[1:]])
    y_surf = np.concatenate([np.flip(y_upper),y_lower[1:]])
    x_surf[0] = 1.0
    x_surf[-1] = 1.0
    y_surf[0] = 0.0   
    y_surf[-1] = 0.0  
    return x_surf, y_surf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="NACA 4-Digit 2D Geometry Generator")
    parser.add_argument("--naca", type=str, default='2412')
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
  
    x, y = generate_naca(args.naca)
    
    if args.output:
        np.savetxt(args.output, np.column_stack([x, y]), delimiter=',', header='x,y', comments='')
    else:
        plt.figure(figsize=(10, 3))
        plt.plot(x, y, color='black', linewidth=1.5)
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    