import numpy as np
import math
import cv2

def to_grayscale(filelist):
    rgb_image_list = []
    gray_image_list = []

    for file in filelist:
        image_rgb = cv2.imread(file)
        rgb_image_list.append(image_rgb)

        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        gray_image_list.append(gray_image)

    return gray_image_list,rgb_image_list

def extract_patches(images, num_patches):
    all_patches = []

    for image in images:
        # Calculate the patch size
        image_height, image_width = image.shape
        patch_height = math.floor(image_height / int(math.sqrt(num_patches)))
        patch_width = math.floor(image_width / int(math.sqrt(num_patches)))
        patch_size = (patch_width, patch_height)

        # Extract patches from the image
        patches = []
        row_step = image_height // int(math.sqrt(num_patches))
        col_step = image_width // int(math.sqrt(num_patches))

        for row in range(0, image_height - patch_height + 1, row_step):
            for col in range(0, image_width - patch_width + 1, col_step):
                patch = image[row:row+patch_height, col:col+patch_width]
                patches.append(patch)

        all_patches.append(patches)

    return np.array(all_patches)

def reconstruct_image(processed_patches, original_shape, num_images):
    original_height, original_width = original_shape

    # Get the patch height and patch width from the processed_patches array
    _, channels, patch_height, patch_width = processed_patches.shape

    # Number of patches along each dimension
    num_patches_height = original_height // patch_height
    num_patches_width = original_width // patch_width

    # Create an empty array to store the reconstructed images
    reconstructed_images = np.zeros((num_images, original_height, original_width), dtype=np.uint8)

    # Iterate over the images
    patch_index = 0
    for img_idx in range(num_images):
        # Iterate over the patches and place them in the reconstructed image
        for row in range(num_patches_height):
            for col in range(num_patches_width):
                # Check if the patch index is within the valid range
                if patch_index >= processed_patches.shape[0]:
                    break

                # Calculate the position of the current patch in the reconstructed image
                row_start = row * patch_height
                row_end = row_start + patch_height
                col_start = col * patch_width
                col_end = col_start + patch_width

                # Copy the processed patch into the reconstructed image
                reconstructed_images[img_idx, row_start:row_end, col_start:col_end] = processed_patches[patch_index, 0]
                patch_index += 1

    return reconstructed_images
