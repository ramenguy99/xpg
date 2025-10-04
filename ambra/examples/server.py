from pyxpg import Action, Key, Modifiers, imgui

from ambra.config import Config, ServerConfig
from ambra.server import Client, MessageId, RawMessage
from ambra.utils.hook import hook
from ambra.viewer import Viewer


class Viewer(Viewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = []

    @hook
    def on_gui(self):
        if imgui.begin("Window")[0]:
            for message in self.messages:
                imgui.text(message)
        imgui.end()

    @hook
    def on_key(self, key: Key, action: Action, modifiers: Modifiers):
        pass

    def on_raw_message(self, client: Client, raw_message: RawMessage):
        print(
            f"Message received from: {client.name} ({client.address}, {client.port}): {raw_message.id} {raw_message.format} {MessageId.USER.value}"
        )
        if raw_message.id == MessageId.USER.value:
            print("Decoding..")
            self.messages.append(raw_message.data.decode("utf-8"))
        else:
            super().on_raw_message(client, raw_message)


config = Config(
    server=ServerConfig(enabled=True),
)
viewer = Viewer("Server", config=config)
viewer.run()
