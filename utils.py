import numpy as np

# Linear interpolation between a and b
def Lerp(a, b, t):
    return (1.0 - t)*a + t*b

def TransferFunc(hsv, k):
    a = 0.0
    b = 1.0
    new_img = np.copy(hsv)
    new_img[:, :, 2] = (a + b*new_img[:, :, 2])**k
    return new_img

def SampleSquare(num_samples, limits):
    pts = []
    for i in range(num_samples):
        x = np.random.uniform(low=limits[0], high=limits[2])
        y = np.random.uniform(low=limits[1], high=limits[3])
        pts.append((x,y))
    return pts


