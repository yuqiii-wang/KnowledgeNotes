import numpy as np
import matplotlib.pyplot as plt

# Define pooling parameters
pool_size = (3, 3)  # Max pooling window size
stride = (3, 3)     # Stride size

# Helper function to add noise and dim pixels
def apply_noise_and_dim(arr, dim_level=30):
    dimmed_arr = arr.copy()
    for i in range(dimmed_arr.shape[0]):
        for j in range(dimmed_arr.shape[1]):
            if dimmed_arr[i][j] > 255:
                dimmed_arr[i][j] -= dim_level
    return dimmed_arr

# Define ReLU function
def relu(matrix):
    return np.maximum(0, matrix)

# Function to perform max pooling
def max_pooling(matrix, pool_size, stride):
    output_height = (matrix.shape[0] - pool_size[0]) // stride[0] + 1
    output_width = (matrix.shape[1] - pool_size[1]) // stride[1] + 1
    pooled_output = np.zeros((output_height, output_width))
    
    for i in range(0, matrix.shape[0] - pool_size[0] + 1, stride[0]):
        for j in range(0, matrix.shape[1] - pool_size[1] + 1, stride[1]):
            pooled_output[i // stride[0], j // stride[1]] = np.max(
                matrix[i:i + pool_size[0], j:j + pool_size[1]]
            )
    return pooled_output


# Function to normalize matrix
def normalize(matrix, target_min=0, target_max=255):
    """
    Normalize the input matrix to a specified range.

    Parameters:
        matrix (np.ndarray): Input matrix (e.g., convolution output).
        target_min (float): Minimum value of the target range.
        target_max (float): Maximum value of the target range.

    Returns:
        np.ndarray: Normalized matrix.
    """
    matrix_min = np.min(matrix)
    matrix_max = np.max(matrix)

    # Avoid division by zero in case all values are the same
    if matrix_max == matrix_min:
        return np.full_like(matrix, target_min)

    # Apply min-max normalization
    normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)  # Scale to [0, 1]
    normalized_matrix = normalized_matrix * (target_max - target_min) + target_min  # Scale to [target_min, target_max]
    return normalized_matrix

# Step 1: Create synthetic MNIST-like images of 6 and 9
def create_synthetic_6():
    image = np.array([
        [0]*28 for _ in range(28)
    ])

    # Draw the digit "6" within the 28x28 grid
    image[1:26, 7:24] = [
        [0,   0,   0,   0,   0,   0,   0,   0,   78,  111, 89,  0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   0,   0,   0,   0,   101, 199, 199, 66,  0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   0,   0,   0,   111, 255, 255, 87,  11,  0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   0,   0,   123, 255, 255, 87,  0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   0,   144, 255, 255, 87,  0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   123, 255, 255, 78,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   0,   113, 189, 255, 82,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   0,   189, 255, 255, 82,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [0,   67,  213, 255, 82,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [62,  233, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [82,  255, 255, 82,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   ],
        [82,  255, 255, 82,  77,  77,  77,  77,  77,  77,  77,  0,   0,   0,   0,   0,   0,   ],
        [82,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 82,  0,   0,   0,   0,   0,   ],
        [82,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 82,  0,   0,   0,   0,   ],
        [82,  255, 255, 77,  43,  0,   0,   0,   0,   177, 255, 255, 82,  82,  0,   0,   0,   ],
        [82,  255, 255, 44,  0,   0,   0,   0,   0,   0,   153, 255, 255, 82,  82,  0,   0,   ],
        [82,  255, 255, 32,  0,   0,   0,   0,   0,   0,   0,   132, 255, 255, 82,  0,   0,   ],
        [82,  255, 255, 11,  0,   0,   0,   0,   0,   0,   0,   92,  255, 255, 82,  0,   0,   ],
        [82,  255, 255, 31,  0,   0,   0,   0,   0,   0,   0,   132, 255, 255, 82,  0,   0,   ],
        [72,  255, 255, 132, 0,   0,   0,   0,   0,   0,   0,   255, 255, 231, 82,  0,   0,   ],
        [75,  255, 255, 211, 112, 0,   0,   0,   0,   0,   201, 255, 255, 82,  0,   0,   0,   ],
        [78,  188, 255, 255, 131, 99,  0,   0,   0,   156, 255, 255, 178, 0,   0,   0,   0,   ],
        [0,   77,  167, 255, 132, 123, 77,  55,  65,  255, 255, 82,  0,   0,   0,   0,   0,   ],
        [0,   0,   82,  255, 255, 255, 255, 255, 255, 255, 82,  0,   0,   0,   0,   0,   0,   ],
        [0,   0,   0,   77,  255, 255, 255, 255, 255, 82,  0,   0,   0,   0,   0,   0,   0,   ],
    ]

    # Add some noise to make it look more realistic
    image = apply_noise_and_dim(image)
    return image

def create_synthetic_9():
    # MNIST-like representation of the number 9 as a 180-degree rotation of digit 6
    digit_6 = create_synthetic_6()
    image = np.rot90(digit_6, 2)
    return image

# Step 2: Define convolutional kernels
def create_diagonal_edge_kernel():
    kernel = np.array([[ 3,  1,  0, -1, -3],
                    [ 1,  3,  1,  0, -1],
                    [ 0,  1,  3,  1,  0],
                    [-1,  0,  1,  3,  1],
                    [-3, -1,  0,  1,  3]]
                )
    kernel = 1/np.sum(kernel) * kernel
    return kernel

def create_rot_diagonal_edge_kernel():
    kernel = np.array([[ 3,  1,  0, -1, -3],
                    [ 1,  3,  1,  0, -1],
                    [ 0,  1,  3,  1,  0],
                    [-1,  0,  1,  3,  1],
                    [-3, -1,  0,  1,  3]]
                )
    kernel = np.rot90(kernel)
    kernel = 1/np.sum(kernel) * kernel
    return kernel

# Step 3: Apply convolution
def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

# Create synthetic images
image_6 = create_synthetic_6()

# Create kernels
diagonal_kernel = create_diagonal_edge_kernel()
rot_diagonal_kernel = create_rot_diagonal_edge_kernel()

# Apply convolution
output_6_diagonal = convolve(image_6, diagonal_kernel)
output_6_rot_diagonal = convolve(image_6, rot_diagonal_kernel)
output_6_diagonal = normalize(output_6_diagonal)
output_6_rot_diagonal = normalize(output_6_rot_diagonal)

# Apply relu
output_6_relu_diagonal = relu(output_6_diagonal)
output_6_relu_rot_diagonal = relu(output_6_rot_diagonal)

# Apply max pooling
output_6_pooled_diagonal = max_pooling(output_6_relu_diagonal, pool_size, stride)
output_6_pooled_rot_diagonal = max_pooling(output_6_relu_rot_diagonal, pool_size, stride)

# Plot the results
fig = plt.figure(figsize=(10, 8))

plt.subplot(3, 3, 4)
plt.title("'6' Image")
plt.imshow(image_6, cmap='gray')

ax1 = fig.add_subplot(3, 3, 1)
ax1.set_title("Diagonal Edge Kernel")
ax1.imshow(diagonal_kernel, cmap='gray', interpolation='nearest')
ax1.axis('off')
# Scale both x and y axes and center the image
scale_factor = 3  # Adjust to shrink or enlarge
image_size = diagonal_kernel.shape[0] / 2  # Half the image size to calculate limits
ax1.set_xlim(-image_size * scale_factor+image_size, image_size * scale_factor+image_size)  # Scale x-axis
ax1.set_ylim(image_size * scale_factor+image_size, -image_size * scale_factor+image_size)  # Scale y-axis (reverse to maintain orientation)


plt.subplot(3, 3, 2)
plt.title("Conv (6)")
plt.imshow(output_6_diagonal, cmap='gray')

plt.subplot(3, 3, 3)
plt.title("Max Pooled (6)")
plt.imshow(output_6_pooled_diagonal, cmap='gray')


ax2 = fig.add_subplot(3, 3, 7)
ax2.set_title("Rotated Diagonal Edge Kernel")
ax2.imshow(rot_diagonal_kernel, cmap='gray', interpolation='nearest')
ax2.axis('off')
# Scale both x and y axes and center the image
scale_factor = 3  # Adjust to shrink or enlarge
image_size = rot_diagonal_kernel.shape[0] / 2  # Half the image size to calculate limits
ax2.set_xlim(-image_size * scale_factor+image_size, image_size * scale_factor+image_size)  # Scale x-axis
ax2.set_ylim(image_size * scale_factor+image_size, -image_size * scale_factor+image_size)  # Scale y-axis (reverse to maintain orientation)


plt.subplot(3, 3, 8)
plt.title("Conv (6)")
plt.imshow(output_6_rot_diagonal, cmap='gray')

plt.subplot(3, 3, 9)
plt.title("Max Pooled (6)")
plt.imshow(output_6_pooled_rot_diagonal, cmap='gray')

plt.tight_layout()
plt.show()
