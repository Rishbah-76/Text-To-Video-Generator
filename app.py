import gradio as gr
import torch

from model import Model, ModelType
from app_canny import create_demo as create_demo_canny
from app_pose import create_demo as create_demo_pose
from app_text_to_video import create_demo as create_demo_text_to_video
from app_pix2pix_video import create_demo as create_demo_pix2pix_video
from app_canny_db import create_demo as create_demo_canny_db
from app_depth import create_demo as create_demo_depth
import argparse
import os

on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"
model = Model(device='cuda', dtype=torch.float16)
parser = argparse.ArgumentParser()
parser.add_argument('--public_access', action='store_true',
                    help="if enabled, the app can be access from a public url", default=False)
args = parser.parse_args()


with gr.Blocks(css='style.css') as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            <a href="#" style="color:yellow;">TEXT TO VIDEO GENERATOR</a> 
        </h1>
        </div>
        """)

    with gr.Tab('Zero-Shot Text2Video'):
        create_demo_text_to_video(model)
    with gr.Tab('Video Instruct Pix2Pix'):
        create_demo_pix2pix_video(model)



if on_huggingspace:
    demo.queue(max_size=20)
    demo.launch(debug=True)
else:
    _, _, link = demo.queue(api_open=False).launch(
        file_directories=['temporal'], share=True)
    print(link)
