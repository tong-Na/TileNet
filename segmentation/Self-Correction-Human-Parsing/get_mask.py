from PIL import Image, ImageDraw
import os


def create_mask(image_path, output_path):
    """
    Create a mask image where the pixels of the specified color in the original image
    are turned black in the mask.

    :param image_path: Path to the original image
    :param target_color: The color to be masked (in RGB format, e.g., (255, 255, 255) for white)
    :param output_path: Path to save the mask image
    """
    # Load the original image
    original = Image.open(image_path)
    width, height = original.size

    # Create a blank image with the same dimensions
    mask = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(mask)

    # Process each pixel
    # print(original.getpixel((400,900)))
    for x in range(width):
        for y in range(height):
            pixel = original.getpixel((x, y))
            if pixel == 5 or pixel == 6 or pixel == 7 or pixel == 0 or pixel == 10:
                draw.point((x, y), fill="black")

    # Save the mask image
    mask.save(output_path)


# Example usage:
# create_mask('D://downloads//Google//TFill-main//TFill-main//00000_00.png', 'D://downloads//Google//TFill-main//TFill-main//0.png')
# This will create a mask from an image, turning all white pixels (255, 255, 255) to black in the mask image.


def list_files(directory):
    """
    Reads the names of the files in the given directory and returns them as a list.
    """
    # List all files and directories in the specified directory
    file_list = os.listdir(directory)
    # Filter out directories, only keep files
    file_list = [
        file for file in file_list if os.path.isfile(os.path.join(directory, file))
    ]
    return file_list


# Replace 'your_directory_path' with the path to the directory you want to list
directory_path = "~/try-on/segmentation/Self-Correction-Human-Parsing/outputs"
files = list_files(directory_path)

index = 0
for i in files:
    print(index)
    index = index + 1
    p1 = "~/try-on/segmentation/Self-Correction-Human-Parsing/outputs/" + i
    # print(p1)
    p2 = "~/try-on/segmentation/Self-Correction-Human-Parsing/masks/" + i
    # print(p2)
    create_mask(p1, p2)
