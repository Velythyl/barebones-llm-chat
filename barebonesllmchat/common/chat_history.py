import copy
import pathlib
from dataclasses import dataclass
import dataclasses
from enum import Enum
from typing import List, Tuple, Union, Dict, Any

from PIL import Image

from barebonesllmchat.common.image_handling import hash_image


# class syntax

class CHAT_ROLE(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3

@dataclass
class ChatHistory:
    history: Tuple = tuple()

    def get_all_image_hashes(self):
        images = [chat["image"] for chat in self.history if chat["image"] is not None]
        return images

    def __getitem__(self, item):
        return self.history[item]

    def __len__(self):
        return len(self.history)

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def add(self, role: Union[CHAT_ROLE, str], message, image=None):
        if isinstance(role, CHAT_ROLE):
            role = role.name

        role = role.lower().capitalize()

        new_message = {"role": role, "content": message, "image": image}
        new_history = (*self.history, new_message)
        return self.replace(history=new_history)

    def add_img(self, image):
        return self.replace(images=(*self.images, image))

    def pack(self) -> str:
        ret = []
        for message in self.history:
            ret.append(f"{message['role'].capitalize()}: {message['content']}")
        
        ret = "\n".join(ret) + " Assistant:"
        return ret
    
    def to_lowercase_roles(self):
        ret = []
        for kv in self.history:
            to_add = {}
            for k, v in kv.items():
                if k == "role":
                    v = v.lower()
                to_add[k] = v
            ret.append(to_add)
        return ChatHistory(ret)

    def to_uppercase_roles(self):
        ret = []
        for kv in self.history:
            to_add = {}
            for k, v in kv.items():
                if k == "role":
                    v = v.capitalize()
                to_add[k] = v
            ret.append(to_add)
        return ChatHistory(ret)
    
    def to_system_as_user(self):
        ret = []
        for kv in self.history:
            to_add = {}
            for k, v in kv.items():
                if k == "role" and v.lower() == CHAT_ROLE.SYSTEM.name.lower():
                    v = CHAT_ROLE.USER.name.lower().capitalize()
                to_add[k] = v
            ret.append(to_add)
        return ChatHistory(ret)
    
    @property
    def history_without_images(self):
        ret = []
        for kv in self.history:
            to_add = {}
            for k, v in kv.items():
                k = k.lower()
                if k == "image":
                    continue
                to_add[k] = v
            ret.append(to_add)
        return ret
    
    def pretty(self):
        ret = []
        for message in self.history:
            ret.append(f"{message['role'].capitalize()}: {message['content']}")

        return "\n\n".join(ret)

    @classmethod
    def from_history_dict(cls, dico):
        self = cls()
        for msg in dico:
            self = self.add(msg["role"], msg["content"])
        return self


@dataclass
class ChatHistoryWithImages:
    chat_history: ChatHistory
    images: Dict[str, Any]  # not actually Any, should be a PIL image

    def open_images(self):
        ret = {}
        for hashed, image_path in self.images.items():
            ret[hashed] = open(image_path, "rb")
        return ret

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def _add_image(self, image: Union[str, pathlib.Path]):
        if isinstance(image, pathlib.Path):
            image = str(image)

        assert isinstance(image, str)

        with open(image, "rb") as f:
            hashed = hash_image(f.read())

        new_dict = copy.deepcopy(self.images)
        new_dict[hashed] = image
        return new_dict, hashed

    def add(self, role: Union[CHAT_ROLE, str], message, image: Union[str, pathlib.Path, None]=None):

        image_hash = None

        if image is not None:
            new_image_dict, image_hash = self._add_image(image)
            self = self.replace(images=new_image_dict)

        self = self.replace(chat_history=self.chat_history.add(role, message, image_hash))
        return self


if __name__ == "__name__":
    ch = ChatHistory()
