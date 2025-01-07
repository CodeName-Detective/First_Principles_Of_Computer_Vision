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



# Fourier Transform
def low_pass_filter(M, N, cutoff_freq):
    H = np.zeros((M, N), dtype=np.float32)
    center_x, center_y = M // 2, N // 2
    for u in range(M):
        for v in range(N):
            # Euclidean Distance
            distance = np.sqrt((u - center_x) ** 2 + (v - center_y) ** 2)
            if distance <= cutoff_freq:
                H[u, v] = 1.0
    return H

def fourier_transform(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    original_shape = image.shape
    pad_width = (
        (0, int(original_shape[0])),
        (0, int(original_shape[1]))
    )
    image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    exponentiation_matrix = (-1) ** (x + y)
    
    image = image * exponentiation_matrix
    
    
    F = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    amplidtude = cv2.magnitude(F[:,:,0], F[:,:,1])
    
    amplidtude = np.log(amplidtude + 1)
    
    H = low_pass_filter(image.shape[0], image.shape[1], 100)
    
    G = F * H[:, :, np.newaxis]
    
    filtered_image = cv2.idft(G)
    
    filtered_image = filtered_image[:,:,0]* exponentiation_matrix
    
    filtered_image = filtered_image[:original_shape[0], : original_shape[1]]
    
    return filtered_image, amplidtude


# Read Image
noisy_image = read_image('Noisy_image.png')


# Filtering in Frequency Domain
guassian_fourier, converted_fourier = fourier_transform(noisy_image)

# Saving Image

save_image('guassian_fourier.png', guassian_fourier, cmap='gray')
save_image('converted_fourier.png', converted_fourier, cmap='gray')