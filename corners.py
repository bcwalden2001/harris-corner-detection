import cv2
import numpy as np

# Loading the image
imagePath = input("Enter the name of the image file: ")
image = cv2.imread(imagePath).astype(np.float64)   # Image can take up to a minute to load

# Splitting image into each of its color channel
blue_channel, green_channel, red_channel = cv2.split(image)

# Computing gradients separately for each channel
grad_x_blue, grad_y_blue = np.gradient(blue_channel)
grad_x_green, grad_y_green = np.gradient(green_channel)
grad_x_red, grad_y_red = np.gradient(red_channel)

# Averaging gradients across all color channels
grad_x = (grad_x_blue + grad_x_green + grad_x_red) / 3.0
grad_y = (grad_y_blue + grad_y_green + grad_y_red) / 3.0

# Computing squared and product terms using average gradients
Ixx = grad_x * grad_x
Iyy = grad_y * grad_y
Ixy = grad_x * grad_y

# Initializing Harris response matrix
harris_response_matrix = np.zeros_like(image[:, :, 0], dtype=np.float64)

# Computing Harris response matrix
k = 0.05
for i in range(2, image.shape[0] - 2):
    for j in range(2, image.shape[1] - 2):
        # Summing all gradients in the window around each pixel
        window_Ixx_sum = np.sum(Ixx[i-1:i+2, j-1:j+2])
        window_Iyy_sum = np.sum(Iyy[i-1:i+2, j-1:j+2])
        window_Ixy_sum = np.sum(Ixy[i-1:i+2, j-1:j+2])

        # Computing determinant and trace of the structure tensor
        determinant = (window_Ixx_sum * window_Iyy_sum) - (window_Ixy_sum ** 2)
        trace = window_Ixx_sum + window_Iyy_sum
        harris_response_matrix[i, j] = determinant - (k * (trace ** 2))

# Normalizing the Harris response values
epsilon = 1e-8  # Small value to prevent division by zero
normalized_matrix = harris_response_matrix.copy()

# Scaling values to a range from 0 to 255
normalized_harris_response = (255 * (normalized_matrix - np.min(normalized_matrix)) /
                              (np.max(normalized_matrix) - np.min(normalized_matrix) + epsilon)).astype(np.uint8)

# Setting a threshold for detection (this is the cornerness value)
threshold = 0.3 * np.max(normalized_harris_response)

# Finding corner positions greater than threshold
corner_positions = np.argwhere(normalized_harris_response > threshold)

# Drawing red circles on detected corners
corner_visualization = image.copy()
for y, x in corner_positions:
    cv2.circle(corner_visualization, (x, y), 3, (0, 0, 255), 1)

# Displaying Harris corners on the image
cv2.imshow("Harris Corners on Image", corner_visualization.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
