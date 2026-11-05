import streamlit as st
from PIL import Image
from AdaAttN.util_functions import tensor2im
from torchvision.transforms import v2
from overall import all_results

content_image, style_image, user_prompt = None, None, None
content_name, style_name = "", ""

to_pil_transform = v2.ToPILImage()

# Initialization
if "results" not in st.session_state:
    st.session_state.results = {}

st.title("Text-guided Neural Style Transfer (NST) on specific objects")

user_prompt = st.text_input("Give your object description", placeholder="Style this cat")

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

col3, col4 = st.columns([2, 3])

with col3:
    st.subheader("Style upload")
    uploaded_style_file = st.file_uploader(
    "Style Image", 
    type=["jpg", "jpeg", "png"]
)

with col4:
    # 2. Check if a file has been uploaded
    if uploaded_style_file is not None:
        st.subheader("Style Image")
        # Open the image using PIL
        style_image = Image.open(uploaded_style_file)    
        style_name = uploaded_style_file.name

        
        # Display the uploaded image
        st.image(
            style_image, 
            caption="Style Image", 
            width=256
        )

if(content_image and style_image and user_prompt):
    print(content_image, style_image)
    stylize_button = st.button("Get results",
                            on_click=all_results,
                            args=(
                                user_prompt,
                                {
                                    "img": content_image,
                                    "name": content_name.split('.')[0]
                                },
                                {
                                    "img": style_image,
                                    "name": style_name.split('.')[0]
                                }
                            ),
                            )

    #st.write(f"Results: {st.session_state.get('results', 'None')}")
    print(st.session_state.get("results", 'None'))
    if(len(st.session_state.results.keys()) == 3):

        cropped, stylized = st.columns([2, 2])

        print(st.session_state.results["cropped_portion"].shape, st.session_state.results["stylized_portion"])


        cropped_numpy = tensor2im(st.session_state.results["cropped_portion"])
        cropped_pil = Image.fromarray(cropped_numpy)

        stylized_numpy = tensor2im(st.session_state.results["stylized_portion"])
        stylized_pil = Image.fromarray(stylized_numpy)

        with cropped:
            st.subheader("Detecting and Extracting the Object")

            st.image(cropped_pil, 
                    caption="Extracted Object Image",
                    width=256)

        with stylized:
            st.subheader("Applying NST module to the portion")

            st.image(stylized_pil,
                    caption="Stylized Object Image", 
                    width=256)

        merged = st.container()

        with merged:
            st.subheader("Merging with the Original Content Image")

            st.image(st.session_state.results["merged"],
                    caption="Final Result Image",
                    width=512)



if(user_prompt):
    print(user_prompt)

if(content_image):
    print(content_image, content_name)

if(style_image):
    print(style_image, style_name)
