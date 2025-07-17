from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Type
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio


@dataclass
class ArtiSplatfactoDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: ArtiSplatfactoDataParser)

@dataclass
class ArtiSplatfactoDataParser(Nerfstudio):
    config: ArtiSplatfactoDataParserConfig