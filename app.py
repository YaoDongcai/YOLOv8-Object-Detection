import streamlit as st
from pathlib import Path
import PIL
from ultralytics import YOLO


# Setting page layout
st.set_page_config(
    page_icon=":red_car:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Introduction
st.title("Object Detection using YOLOv8")




# Title


# Sidebar
st.sidebar.header("Model Configuration")

# Model Options
model_type = st.sidebar.radio(
    "--------", ['Object Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100


if model_type == 'Object Detection':
    model_path = Path('..\model\yolov8n.pt')


# Load Pre-trained ML Model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

source_img = None
source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is None:
            default_image_path = str('..\Image\image1.jpg')
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Default Image",
                        use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image",
                        use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    if source_img is None:
        
        default_detected_image_path = str('..\Image\image1d.jpg')
        default_detected_image = PIL.Image.open(
            default_detected_image_path)
        st.image(default_detected_image_path, caption='Detected Image',
                    use_column_width=True)
    else:
        if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image,
                                conf=confidence
                                )
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                        use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.write("No image is uploaded yet!")


st.write("""
YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system that is incredibly fast and accurate. 
YOLOv8 is the latest iteration of this series, featuring several improvements in speed and accuracy over its predecessors.
""")

st.header("Key Features of YOLOv8")
st.write("""
1. **Speed and Efficiency**: YOLOv8 can process images in real-time, making it suitable for applications where speed is critical.
2. **High Accuracy**: Improved detection accuracy with advanced architecture and optimization techniques.
3. **Versatility**: Capable of detecting a wide range of objects in various environments.
4. **Scalability**: Can be scaled to run on different hardware, from high-end GPUs to mobile devices.
5. **Easy Integration**: YOLOv8 can be easily integrated into existing systems and frameworks.
""")

# Applications
st.header("Applications")
st.write("""
YOLOv8 can be used in a variety of applications, including but not limited to:
- Autonomous Vehicles: Detecting pedestrians, other vehicles, and obstacles.
- Surveillance: Monitoring public spaces for safety and security.
- Retail: Analyzing customer behavior and preventing theft.
- Healthcare: Assisting in medical imaging and diagnostics.
- Robotics: Enabling robots to understand and interact with their environment.
""")

# Conclusion
st.header("Conclusion")
st.write("""
YOLOv8 represents a significant advancement in the field of object detection, offering a balance of speed and accuracy that makes it ideal for real-world applications. Whether you're working on cutting-edge research or practical implementations, YOLOv8 provides the tools and performance needed to achieve your goals.
""")

# Footer
st.write("Developed by Akash Kamble. For more information, visit the official [YOLOv8 GitHub Repository](https://github.com/ultralytics/yolov5).")

