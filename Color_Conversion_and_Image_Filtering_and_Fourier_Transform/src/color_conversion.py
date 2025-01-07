import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def save_image(file_name, image_array, cmap=None):
    if cmap:
        plt.imsave(file_name, image_array, cmap=cmap)
    else:
        plt.imsave(file_name, image_array)


# RGN to HSV 1

def rgb_to_hsv_1(image):
    image = image/255
    
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    # Constructing Value
    V = np.maximum(R, np.maximum(G, B))
    
    # Constructing Saturation
    S = np.where(V != 0, (V - np.minimum(R, np.minimum(G, B))) / V, 0)

    
    # Constructing Hue
    H = np.zeros_like(V)
    
    R_mask = (V == R)
    H[R_mask] = 60 * (G[R_mask] - B[R_mask]) / (V[R_mask] - np.minimum(R[R_mask], np.minimum(G[R_mask], B[R_mask])))
    
    G_mask = (V == G)
    H[G_mask] = 60 * (B[G_mask] - R[G_mask]) / (V[G_mask] - np.minimum(R[G_mask], np.minimum(G[G_mask], B[G_mask])) + 120)
    
    B_mask = (V == B)
    H[B_mask] = 60 * (R[B_mask] - G[B_mask]) / (V[B_mask] - np.minimum(R[B_mask], np.minimum(G[B_mask], B[B_mask])) + 240)

    H[H < 0] += 360
    
    V = (V * 255).astype(np.uint8)
    S = (S * 255).astype(np.uint8)
    H = (H / 2).astype(np.uint8)
    
    hsv_image = np.dstack((H,S,V))
    return hsv_image


# RGN to HSV 2

def rgb_to_hsv_2(image):
    
    image = image/255
    
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    V = (R+G+B)/3
    
    S = 1 - ((1/V)*np.minimum(R, np.minimum(G, B)))

    
    H = np.zeros_like(V)

    
    theta = np.arccos(0.5*((R-G)+(R-B))/np.sqrt(np.square(R-G)+((R-B)*(G-B))))
    
    H = np.where(B <= G, theta, 360-theta)
    
    hsv_image = np.dstack(((H).astype(np.uint8),(S * 255).astype(np.uint8),(V * 255).astype(np.uint8)))
    cv2.imwrite('hsv_image_2.png', hsv_image)
    return hsv_image


# RGB to CMYK

def rgb_to_cmyk(image):
    image = image/255
    
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    C = 1-R
    M = 1-G
    Y = 1-B
    
    K = np.minimum(C, np.minimum(M,Y))
    
    C = np.where(K==1, 0, (C-K)/(1-K))
    M = np.where(K==1, 0, (M-K)/(1-K))
    Y = np.where(K==1, 0, (Y-K)/(1-K))
    
    
    cmyk_image = np.dstack(((C*255).astype(np.uint8), (M*255).astype(np.uint8), (Y*255).astype(np.uint8), (K*255).astype(np.uint8)))
    
    return cmyk_image


# RGB To Lab Image

def f(x):
    if x>0.008856:
        return np.cbrt(x)
    else:
        return (7.787*x)+(16/116)

def rgb_to_lab(image):
    image = image/255
    constant_matrix = np.array([[0.412453, 0.357580, 0.180423],[0.212671, 0.715160, 0.072169],[0.019334, 0.119193, 0.950227]])
    
    xyz_matrix = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            xyz_matrix[i, j, :] = np.matmul(constant_matrix, image[i, j, :])

    
    X = xyz_matrix[: ,:, 0]/0.950456
    Y = xyz_matrix[: ,:, 1]
    Z = xyz_matrix[:, :, 2]/1.088754 
    
    
    L = np.where(Y > 0.008856, (116 * np.cbrt(Y)) - 16, 903.3 * Y)
    a = 500*(np.vectorize(f)(X) - np.vectorize(f)(Y)) + 128
    b = 200*(np.vectorize(f)(Y) - np.vectorize(f)(Z)) + 128
    
    lab_image = np.dstack(((L).astype(np.uint8),(a).astype(np.uint8),(b).astype(np.uint8)))
    
    return lab_image



# Reading Image
lenna_rgb_image = read_image('Lenna.png')


# Colour Conversion
hsv_image_1 = rgb_to_hsv_1(lenna_rgb_image)
hsv_image_2 = rgb_to_hsv_2(lenna_rgb_image)
cmyk_image = rgb_to_cmyk(lenna_rgb_image)
lab_image = rgb_to_lab(lenna_rgb_image)

# Saving Images
save_image('hsv_image_1.png', hsv_image_1)
save_image('hsv_image_2.png', hsv_image_2)
save_image('cmyk_image.png', cmyk_image)
save_image('lab_image.png', lab_image)