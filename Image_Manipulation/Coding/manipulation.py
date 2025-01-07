import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading Image
rbf_image = cv2.imread('image.png')
print("shape of image: {0}".format(rbf_image.shape))

# Writing Image Function
def image_write(img, image_name):
    cv2.imwrite(image_name, img)


### Gray scaling an image
def grey_scale_conv(img):
    # gray = 0.2989*red + 0.5870*green + 0.1140*blue
    gray_scale = np.dot(img, np.array([0.2989, 0.5870, 0.1140]))
    return gray_scale

gray_scale_img = grey_scale_conv(rbf_image)
print("shape of scaled image: {0}".format(gray_scale_img.shape))
image_write(gray_scale_img, 'gray_image.png')


### Scaling
def scaling(img):
    #Selecting alternate rows and columns
    scaled_img = img[::2,::2]
    return scaled_img

scaled_img = scaling(gray_scale_img)
image_write(scaled_img, "gray_image_scaled.png")


### Translating
def translating(img):
    translate_image = np.full_like(img, 255)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            try:
                translate_image[row+50, col+50] = img[row, col]
            except IndexError:
                break
    
    return translate_image 

translate_image = translating(gray_scale_img)
image_write(translate_image, "gray_image_translated.png")


### Flip Horizontal
def flip_horizontal(img):
    hor_flipped = np.full_like(img, 0)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            hor_flipped[img.shape[0]-1-row,col] = img[row, col]
    return hor_flipped

hor_flipped_image = flip_horizontal(gray_scale_img)
image_write(hor_flipped_image, "gray_image_flip_horizontal.png")


### Flip Vertical
def flip_vertical(img):
    ver_flipped = np.full_like(img, 0)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            ver_flipped[row,img.shape[1]-1-col] = img[row, col]
    return ver_flipped

ver_flipped_image = flip_vertical(gray_scale_img)
image_write(ver_flipped_image, "gray_image_flip_vertical.png")


### Inversion
def inversion(img):
    inv_image = np.full_like(img, 0)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            inv_image[row,col] = 255-img[row, col]
    return inv_image

inv_image = inversion(gray_scale_img)
image_write(inv_image, "gray_image_inversion.png")


### Rotation
def rotation(img):
    rot_image = np.full_like(img, 0)
    
    angle = np.radians(45)
    
    # Center of image in coordinate axis
    
    x_center = -int(img.shape[1]/2)
    y_center = -int(img.shape[0]/2)
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            
            new_x = int((col+x_center)*np.cos(angle) - (row+y_center)*np.sin(angle)) - x_center
            new_y = int((col+x_center)*np.sin(angle) + (row+y_center)*np.cos(angle)) - y_center
            
            if 0 <= new_x < img.shape[1] and 0 <= new_y < img.shape[0]:
                rot_image[new_y,new_x] = img[row, col]
                
    return rot_image

rot_image = rotation(gray_scale_img)
image_write(rot_image, "gray_image_rotated.png")

# ***************************************** Bonus ******************************** #

### Scaling
def bonus_rbf_scaling(img):
    #Selecting alternate rows and columns
    scaled_img = img[::2,::2]
    return scaled_img

scaled_img_rbf = bonus_rbf_scaling(rbf_image)
image_write(scaled_img_rbf, "image_scaled.png")


### Translating
def bonus_rbf_translating(img):
    translate_image = np.full_like(img, 255)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            try:
                translate_image[row+50, col+50] = img[row, col]
            except IndexError:
                break
    
    return translate_image

translating_img_rbf = bonus_rbf_translating(rbf_image)
image_write(translating_img_rbf, "image_translated.png")


### Horizantal Flip
def flip_horizontal_rbf(img):
    hor_flipped = np.full_like(img, 0)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            hor_flipped[img.shape[0]-1-row,col] = img[row, col]
    return hor_flipped

hor_flip_img_rbf = flip_horizontal_rbf(rbf_image)
image_write(hor_flip_img_rbf, "image_flip_horizontal.png")


### Vertical Flip
def flip_vertical_rbf(img):
    ver_flipped = np.full_like(img, 0)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            ver_flipped[row,img.shape[1]-1-col] = img[row, col]
    return ver_flipped

ver_flipped_image_rbf = flip_vertical_rbf(rbf_image)
image_write(ver_flipped_image_rbf, "image_flip_vertical.png")