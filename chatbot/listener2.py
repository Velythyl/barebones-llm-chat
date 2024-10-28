import json
import os
import shutil
import time

from PIL import Image

import numpy as np
import socketio
import requests

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
import common
from common.chat_history import ChatHistory

CONST_SERVER_IP = '127.0.0.1'

sio = socketio.Client()
sio.connect(f'http://{CONST_SERVER_IP}:5000')

CONST_DOWNLOADS_TO_KEEP = 10
CONST_DOWNLOAD_DIR = "./downloads"
shutil.rmtree(CONST_DOWNLOAD_DIR, ignore_errors=True)   # todo maybe not wipe downloads every init
os.makedirs(CONST_DOWNLOAD_DIR, exist_ok=True)

from chatbot.molmo_bot import Olmo
LLM = Olmo()

@sio.on('new_message_from_user')
def message_event(data):
    # gets called whe
    print("I received")
    print(data)

    #data = json.loads(data)
    chat_id = data['chat_id']
    chat = ChatHistory(tuple(json.loads(data["chat_history"])))

    traverse_and_download_images(chat)
    images = traverse_and_get_images(chat)

    if len(images) == 0:
        images = None

    new_chat = LLM.respond(chat, images)

    send_message(chat_id, "assistant", new_chat.history[-1]["content"], "your_api_key")


def send_message(chat_id, role, message, api_key):
    """Send a response message to the user via the home server, with optional image."""
    # Prepare the form data
    form_data = {
        "chat_id": chat_id,
        "api_key": api_key,
        "role": role,
        "message": message,
    }

    # Prepare the image data if an image is provided

    response = requests.post(
        f"http://{CONST_SERVER_IP}:5000/send_message",
        data=form_data,
        files=None
    )
    if response.status_code == 200:
        print(f"Message sent to chat {chat_id}")
    else:
        print(f"Failed to send message to chat {chat_id}: {response.status_code}")


def scrub_downloads():
    list_of_files = os.listdir(CONST_DOWNLOAD_DIR)
    paths = [f"{CONST_DOWNLOAD_DIR}/{x}" for x in list_of_files]

    if len(paths) < 10:
        return

    #times = np.array(list(map(os.path.getctime, paths)))



    #if len(list_of_files) == 25:
    #    oldest_file = min(full_path, key=os.path.getctime)
    #    os.remove(oldest_file)

def traverse_and_download_images(chat_history):
    images = chat_history.get_all_image_hashes()

    def download_image_if_missing(image_hash):
        """Downloads image from the server if it's missing locally."""
        image_path = os.path.join(CONST_DOWNLOAD_DIR, f"{image_hash}")

        if not os.path.exists(image_path):
            print(f"Image {image_hash} not found locally. Downloading...")
            response = requests.get(f"http://{CONST_SERVER_IP}:5000/get_image/{image_hash}")

            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                print(f"Image {image_hash} downloaded and saved.")
            else:
                print(f"Failed to download image {image_hash}: {response.status_code}")
        else:
            print(f"Image {image_hash} already exists locally.")

    for image in images:
        download_image_if_missing(image)

def traverse_and_get_images(chat_history):
    images = chat_history.get_all_image_hashes()

    images = [Image.open(f"{CONST_DOWNLOAD_DIR}/{image}") for image in images]
    return images


if __name__ == '__main__':
    sio.wait()
