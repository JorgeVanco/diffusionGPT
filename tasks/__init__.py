from .smoltalk import SmolTalkTask
from .everyday import EverydayConversationsTask
from .nemotron import NemotronTask

TASK_REGISTRY = {
    "smoltalk": SmolTalkTask,
    "everyday": EverydayConversationsTask,
    "nemotron": NemotronTask,
}

def get_task(name):
    return TASK_REGISTRY[name]