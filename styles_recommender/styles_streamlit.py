import streamlit as st
from PIL import Image
from helpful_functions import find_evaluation_metric, recommend_label
from torchvision.transforms import v2

content_image, user_prompt = None, None
content_name, style_name = "", ""


# Initialization
if "recommendations" not in st.session_state:
    st.session_state.recommendations = ""

st.title("Styles Recommender")

user_prompt = st.text_input("Give your description on which aspect to focus", placeholder="Color/Structure/Semantic")

col1, col2 = st.columns([2,3])


with col1:
    st.subheader("Content Upload")
    # 1. Create the file uploader widget restricted to images
    uploaded_content_file = st.file_uploader(
        "Content Image", 
        type=["jpg", "jpeg", "png"]
    )

with col2:
    # 2. Check if a file has been uploaded
    if uploaded_content_file is not None:
        st.subheader("Content Image")
        # Open the image using PIL
        content_image = Image.open(uploaded_content_file)
        content_name = uploaded_content_file.name

        
        # Display the uploaded image
        st.image(
            content_image, 
            caption="Content Image", 
            width=256
        )

if(content_image and  user_prompt):
    eval_metric = find_evaluation_metric(user_prompt)
    recommender_button = st.button("Get recommended Label",
                            on_click=recommend_label,
                            args=(
                                eval_metric,
                            )
                            )

    #st.write(f"Results: {st.session_state.get('results', 'None')}")
    print(st.session_state.get("recommendations", 'None'))
    if(st.session_state.recommendations != ""):
        st.write(f"Recommended Label: {st.session_state.recommendations}")