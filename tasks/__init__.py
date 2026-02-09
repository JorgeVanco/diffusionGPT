from .everyday import EverydayConversationsTask
from .nemotron import NemotronTask
from .smoltalk import SmolTalkTask

TASK_REGISTRY = {
    "smoltalk": SmolTalkTask,
    "everyday": EverydayConversationsTask,
    "nemotron": NemotronTask,
}

def get_task(name):
    return TASK_REGISTRY[name]
