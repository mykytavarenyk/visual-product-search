from PIL import Image, UnidentifiedImageError
import io

import io
from PIL import Image, UnidentifiedImageError, ImageOps

def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image file.
    Steps:
      - Load image from uploaded file.
      - Correct orientation based on EXIF data if available.
      - Convert image to RGB.
      - Resize image to a target size (e.g., 224x224) while preserving aspect ratio.
      - Apply auto contrast enhancement.
    """
    try:
        # Load image from file
        image = Image.open(io.BytesIO(uploaded_file.read()))

        # Auto-orient image using EXIF data (if available)
        try:
            exif = image._getexif()
            if exif is not None:
                orientation_key = 274  # CF. EXIF Orientation tag
                if orientation_key in exif:
                    orientation = exif[orientation_key]
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        except Exception:
            # If EXIF orientation data is not available or processing fails, skip correction.
            pass

        # Convert to RGB if not already in that mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image: maintain aspect ratio by using thumbnail, then paste on a blank canvas if needed.
        target_size = (224, 224)
        image.thumbnail(target_size, Image.ANTIALIAS)

        # Create a new image with a white background and target size,
        # then paste the resized image centered on it.
        new_image = Image.new("RGB", target_size, (255, 255, 255))
        offset = ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2)
        new_image.paste(image, offset)

        # Apply auto contrast enhancement
        new_image = ImageOps.autocontrast(new_image)

        return new_image
    except UnidentifiedImageError:
        raise ValueError("The uploaded file is not a valid image.")
    except Exception as e:
        raise ValueError(f"An error occurred during image preprocessing: {e}")

