from dataclasses import dataclass

@dataclass
class ColorInfo:
    face_color: str
    border_color: str

YELLOW = ColorInfo(face_color="#FFF2CC", border_color="#D6B656")
GREEN = ColorInfo(face_color="#D5E8D4", border_color="#82B366")
VIOLET = ColorInfo(face_color="#E1D5E7", border_color="#9673A6")
BLUE = ColorInfo(face_color="#DAE8FC", border_color="#6C8EBF")
ORANGE = ColorInfo(face_color="#FFE6CC", border_color="#D79B00")
