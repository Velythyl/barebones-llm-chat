import atexit
import functools
import json
import os
import hashlib
import shutil
import signal

from flask_socketio import SocketIO, emit

from flask import Flask, request, jsonify, send_from_directory
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import dataclasses
from typing import Dict, Tuple, Union


import sys
import pathlib

from barebonesllmchat.server.random_names import generate_name

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from barebonesllmchat.common.chat_history import CHAT_ROLE, ChatHistory
from barebonesllmchat.common.image_handling import save_image



# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Configure the image storage directory
secrets = json.load(open(pathlib.Path(__file__).parent.parent / "secrets" / 'secrets.json'))
UPLOAD_FOLDER = secrets["server_upload_dir"] # './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Class and chat history definitions (reuse your existing classes)



# In-memory store for chat histories
chats: Dict[str, ChatHistory] = {}

def chat_names():
    return list(chats.keys())


CONST_CHAT_SAVE_PATH = secrets["server_save_chat_path"]

def graceful_bootup():
    global chats
    if os.path.exists(CONST_CHAT_SAVE_PATH):
        with open(CONST_CHAT_SAVE_PATH, "r") as jsonfile:
            temp_chats = json.load(jsonfile)

        temp_chats = {chat_name: ChatHistory(tuple(history)) for chat_name, history in temp_chats.items()}
        chats = temp_chats
graceful_bootup()

def graceful_shutdown(signum, frame):
    global chats
    temp_chats = {chat_name: chat.history for chat_name, chat in chats.items()}
    with open(CONST_CHAT_SAVE_PATH, "w") as jsonfile:
        jsonned = json.dumps(temp_chats)
        jsonfile.write(jsonned)
    exit()

# Register the graceful shutdown function for SIGTERM and SIGINT
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# Also ensure cleanup happens when the process exits
#atexit.register(functools.partial(graceful_shutdown, None, None))

def authenticate(api_key):
    return api_key in secrets["valid_api_keys"]


# WebSocket route to notify chatbot
@socketio.on('connect')
def handle_connect():
    print("Chatbot connected to WebSocket")

def notify_new_message(chat_id, generation_settings):
    # Broadcasts to all connected WebSocket clients (e.g., the chatbot)
    jsonned = json.dumps(chats[chat_id].history)
    generation_settings = json.dumps(generation_settings)

    socketio.emit('new_message_from_user', {'chat_id': chat_id, 'chat_history': jsonned, "generation_settings": generation_settings})

# Create a new chat
@app.route('/create_chat', methods=['POST'])
def create_chat():
    data = request.form.to_dict()
    api_key = data.get('api_key')
    if not authenticate(api_key):
        return jsonify({"error": "Unauthorized"}), 403

    #chat_id = data.get('chat_id', None)
    #if chat_id is None:
    chat_id = generate_name(chat_names())

    chats[chat_id] = ChatHistory()
    return jsonify({"chat_id": chat_id}), 201


# Get all chat IDs
@app.route('/get_chats', methods=['GET'])
def get_chats():
    return jsonify(list(chats.keys()))

def notify_client_got_message(chat_id):
    socketio.emit('new_message_from_assistant', {"chat_id": chat_id})

# Send a message to a chat, including image handling
@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.form.to_dict()
    chat_id = data.get('chat_id')
    api_key = data.get('api_key')
    role = data.get('role')
    message = data.get('message')
    image = request.files.get('image')

    generation_settings = data.get("generation_settings", {})

    if not authenticate(api_key):
        return jsonify({"error": "Unauthorized"}), 403

    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404

    image_hash = None
    if image:
        # Save the image and get its hash
        image_hash = save_image(image)

    # Add the message (with image hash if applicable)
    chats[chat_id] = chats[chat_id].add(role, message, image=image_hash)

    if role.lower() == CHAT_ROLE.USER.name.lower():
        notify_new_message(chat_id, generation_settings)
    elif role.lower() == CHAT_ROLE.ASSISTANT.name.lower():
        notify_client_got_message(chat_id)

    return jsonify({"status": "Message added"}), 200

@app.route('/send_history', methods=['POST'])
def send_history():
    data = request.form.to_dict()
    chat_id = data.get('chat_id')
    api_key = data.get('api_key')
    chat_history = ChatHistory(tuple(json.loads(data.get('chat_history'))))
    generation_settings = data.get("generation_settings", {})

    if not authenticate(api_key):
        return jsonify({"error": "Unauthorized"}), 403

    chats[chat_id] = chat_history

    if len(chat_history.history) == 0:
        notify_client_got_message(chat_id)
        return jsonify({"status": "Chat History imported, assistant NOT notified because history len is 0"}), 200


    images = request.files #.getall() # {hash: bytes}
    for image_hash, image in images.items():
        if os.path.exists(os.path.join(UPLOAD_FOLDER, image_hash)):
            continue
        else:
            save_image(image, image_hash)


    role = chat_history.history[-1]["role"]
    assert role.lower() in [CHAT_ROLE.USER.name.lower(), CHAT_ROLE.SYSTEM.name.lower()]
    notify_new_message(chat_id, generation_settings)

    return jsonify({"status": "Chat History imported, assistant notified"}), 200


# Get the history of a specific chat (returns image hashes)
@app.route('/get_chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404

    return jsonify(chats[chat_id].history)


# Delete a specific chat
@app.route('/delete_chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    data = request.form.to_dict()
    api_key = data.get('api_key')
    if not authenticate(api_key):
        return jsonify({"error": "Unauthorized"}), 403

    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404

    def get_images_for_chat(key):
        cur_chat = chats[key].history
        images = [chat["image"] for chat in cur_chat if chat["image"] is not None]
        return images

    images_of_all_chats = {key: get_images_for_chat(key) for key in chats}

    unsafe_images = []
    chat_id_images = images_of_all_chats[chat_id]
    for image in chat_id_images:
        for key, other_chat_images in images_of_all_chats.items():
            if key == chat_id:
                continue
            if image in other_chat_images:
                unsafe_images.append(image)

    safe_images = set(images_of_all_chats[chat_id]) - set(unsafe_images)

    for safe_image in safe_images:
        os.remove(os.path.join(UPLOAD_FOLDER, safe_image))

    del chats[chat_id]
    return jsonify({"status": "Chat deleted"}), 200


# Get chats where the latest message is from the user (for the cluster)
@app.route('/get_new_messages', methods=['GET'])
def get_new_messages():
    user_chats = {chat_id: chat.history for chat_id, chat in chats.items()
                  if chat.history and chat.history[-1]["role"] == "USER"}

    return jsonify(user_chats)


# Retrieve an image by its hash
@app.route('/get_image/<image_hash>', methods=['GET'])
def get_image(image_hash):
    image_path = os.path.join(UPLOAD_FOLDER, image_hash)
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404

    return send_from_directory(UPLOAD_FOLDER, image_hash)

@app.route('/')
def serve_homepage():
    return send_from_directory('static', 'index.html')


# Run the Flask app
if __name__ == '__main__':

    try:
        socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True, use_reloader=False)
    except KeyboardInterrupt:
        # This except block will handle a keyboard interrupt (Ctrl+C)
        graceful_shutdown(None, None)

