import cv2

def load_and_scale_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0
    return scaled_gray_image
