from PIL import Image, UnidentifiedImageError
import io

def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image file.
    Supports various formats and performs basic validation.
    """
    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        # Convert image to RGB if not already in that mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except UnidentifiedImageError:
        raise ValueError("The uploaded file is not a valid image.")
    except Exception as e:
        raise ValueError(f"An error occurred during image preprocessing: {e}")
