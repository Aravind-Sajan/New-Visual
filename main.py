from PIL import Image
import matplotlib.pyplot as plt

# Define the path to your sign language alphabet images folder
images_folder = r'C:\Users\aravi\Desktop\Hackathon Jan\Alpha\\'

# Define a mapping from text to image filenames
sign_language_dict = {
    'A': 'A.png',
    'B': 'B.png',
    'C': 'C.png',
    'D': 'D.png',
    'E': 'E.png',
    'F': 'F.png',
    'G': 'G.png',
    'H': 'H.png',
    'I': 'I.jpg',
    'J': 'J.jpg',
    'K': 'K.jpg',
    'L': 'L.jpg',
    'M': 'M.jpg',
    'N': 'N.jpg',
    'O': 'O.jpg',
    'P': 'P.jpg',
    'Q': 'Q.jpg',
    'R': 'R.jpg',
    'S': 'S.jpg',
    'T': 'T.jpg',
    'U': 'U.jpg',
    'V': 'V.jpg',
    'W': 'W.jpg',
    'X': 'X.jpg',
    'Y': 'Y.jpg',
    'Z': 'Z.jpg',
    ' ': 'SPACE.jpeg'
}
# Function to translate text to sign language images
def text_to_sign_language(text):
    # Create a list to store the sign language images
    sign_language_images = []

    for letter in text.upper():
        if letter in sign_language_dict:
            image_path = images_folder + sign_language_dict[letter]
            try:
                img = Image.open(image_path)
                sign_language_images.append(img)
            except FileNotFoundError:
                print(f"Image not found for letter: {letter}")

    return sign_language_images
text_to_translate = input("Enter text: ")
translated_images = text_to_sign_language(text_to_translate)

# Concatenate and display the translated images using matplotlib
if translated_images:
    plt.figure(figsize=(12, 4))
    total_width = sum(img.width for img in translated_images)
    max_height = max(img.height for img in translated_images)
    combined_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in translated_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    plt.imshow(combined_image)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()
