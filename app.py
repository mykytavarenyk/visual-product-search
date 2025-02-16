import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from utils.image_utils import preprocess_image
from models.clip_model import get_embedding
from utils.similarity_search import find_similar_products

def load_css(file_name):
    """Load external CSS file."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Visual Product Search",
        page_icon=":mag:",
        layout="wide"
    )

    # Load the new dark theme CSS
    load_css("static/styles.css")

    st.markdown("<div class='header-title'>Visual Product Search</div>", unsafe_allow_html=True)
    st.markdown("**Upload an image, crop if needed, and search for similar products.**")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a product image", 
            type=["jpg", "jpeg", "png", "bmp", "gif"]
        )
    with col2:
        st.markdown("### Instructions")
        st.markdown(
            """
            1. Upload an image of a product.
            2. Optionally, crop the image using the checkbox below.
            3. Click **Search Similar Products** to view matching items.
            """
        )

    if uploaded_file is not None:
        try:
            image = preprocess_image(uploaded_file)
            with st.expander("View Original Image"):
                st.image(image, caption="Uploaded Image", use_column_width=True)

            crop_option = st.checkbox("Crop Image")
            if crop_option:
                st.markdown("### Crop Your Image")
                cropped_image = st_cropper(image, realtime_update=True)
            else:
                cropped_image = image

            if crop_option:
                st.markdown("### Preview")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(image, caption="Original Image", use_column_width=True)
                with col_b:
                    st.image(cropped_image, caption="Cropped Image", use_column_width=True)
            else:
                st.image(cropped_image, caption="Image for Processing", use_column_width=True)

            query_embedding = get_embedding(cropped_image)

            filter_option = st.selectbox(
                "Filter results by file extension", 
                options=["All", ".jpg", ".jpeg", ".png", ".bmp", ".gif"]
            )

            if st.button("Search Similar Products"):
                with st.spinner("Searching..."):
                    results = find_similar_products(query_embedding, top_n=5, filter_extension=filter_option)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<div class='subheader'>Similar Products</div>", unsafe_allow_html=True)
                
                if results:
                    cols = st.columns(5)
                    for i, result in enumerate(results):
                        with cols[i]:
                            st.image(result['image_path'], use_column_width=True)
                            st.markdown(f"**Score:** {result['score']:.2f}")
                else:
                    st.error("No similar products found.")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
    else:
        st.info("Please upload an image file.")

if __name__ == "__main__":
    main()
