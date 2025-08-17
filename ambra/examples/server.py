from pyxpg import Action, Key, Modifiers, imgui

import ambra
from ambra.config import Config
from ambra.server import Client, MessageId, RawMessage
from ambra.utils.hook import hook


class Viewer(ambra.Viewer):
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
    server_enabled=True,
)
viewer = Viewer("Hello World", 1280, 720, config=config)
viewer.run()
