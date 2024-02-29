import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = YOLO('best.pt')

uploaded_files = st.file_uploader("Выберите изображения", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
    result = model.predict(image, device=device)[0]
    result.save(uploaded_file.name)
    filename = f'{uploaded_file.name[:-4]}.txt'
    result.save_txt(filename)
    st.image(uploaded_file.name, caption=uploaded_file.name)
    st.download_button(
        label="Скачать",
        data=open(filename, 'r'),
        file_name=filename
    )
    os.remove(uploaded_file.name)
