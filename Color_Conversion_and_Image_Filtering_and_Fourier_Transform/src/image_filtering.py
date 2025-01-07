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


# Convolution 
def convolution(image, filter, padding = 0):
    if padding:
        image = np.pad(image, pad_width=((padding, padding), (padding, padding), (0,0)), mode='constant', constant_values=0)
    convolved_image = image.copy()
    
    filter = np.flipud(np.fliplr(filter))
    for j in range(image.shape[1] - filter.shape[1] + 1):
        for i in range(image.shape[0] - filter.shape[0] + 1):
            convolved_value = 0
            for y_idx in range(filter.shape[1]):
                for x_idx in range(filter.shape[0]):
                    convolved_value += image[i + x_idx, j + y_idx] * filter[x_idx, y_idx]
            convolved_image[i, j, :] = convolved_value
    if padding:
        convolved_image = convolved_image[padding:-padding,padding:-padding, :]
    return convolved_image



# Filtering
def filtering(image, filter, padding = 0):
    if padding:
        image = np.pad(image, pad_width=((padding, padding), (padding, padding), (0,0)), mode='constant', constant_values=0)
    filtered_image = image.copy()
    
    for j in range(image.shape[1] - filter.shape[1] + 1):
        for i in range(image.shape[0] - filter.shape[0] + 1):
            filtered_value = 0
            for y_idx in range(filter.shape[1]):
                for x_idx in range(filter.shape[0]):
                    filtered_value += image[i + x_idx, j + y_idx] * filter[x_idx, y_idx]
            filtered_image[i+int(np.floor(filter.shape[0]/2)), j+int(np.floor(filter.shape[1]/2)), :] = filtered_value
    if padding:
        filtered_image = filtered_image[padding:-padding,padding:-padding, :]
    return filtered_image


# Median Filtering
def median_filtering(image, filter_shape = (5,5), padding = 0):
    if padding:
        image = np.pad(image, pad_width=((padding, padding), (padding, padding), (0,0)), mode='constant', constant_values=0)
    filtered_image = image.copy()
    
    for j in range(image.shape[1] - filter_shape[1] + 1):
        for i in range(image.shape[0] - filter_shape[0] + 1):
            filtered_values = []
            for y_idx in range(filter_shape[1]):
                for x_idx in range(filter_shape[0]):
                    filtered_values.append(image[i + x_idx, j + y_idx])
            filtered_image[i+int(np.floor(filter_shape[0]/2)), j+int(np.floor(filter_shape[1]/2)), :] = np.median(np.array(filtered_values), axis=0)
    if padding:
        filtered_image = filtered_image[padding:-padding,padding:-padding, :]
    return filtered_image


# Brightness and Contrast Adjustment
def brightness_contrast_adjustment(image_array, brightness, contrast):
    adjusted_image = np.clip(contrast * image_array + brightness, 0, 255).astype(np.uint8)
    return adjusted_image



# Read Image
noisy_image = read_image('Noisy_image.png')
uexposed_image = read_image('Uexposed.png')


# Filters
mask = 1/9 * np.ones((3,3))
gaussian_mask = 1/16 * np.array([[1,2,1],[2,4,2],[1,2,1]])



# Filtering
convolved_image = convolution(noisy_image, mask, padding=1)
average_image = filtering(noisy_image, mask, padding=1)
gaussian_image = filtering(noisy_image, gaussian_mask, padding=1)
median_image = median_filtering(noisy_image, filter_shape = (5,5), padding = 1)

adjusted_image = brightness_contrast_adjustment(uexposed_image, brightness=100, contrast=1.8)


# Saving Image

save_image('convolved_image.png', convolved_image)
save_image('average_image.png', average_image)
save_image('gaussian_image.png', gaussian_image)
save_image('median_image.png', median_image)
save_image('adjusted_image.png', adjusted_image)