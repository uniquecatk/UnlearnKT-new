import json
import re

try:
    from jsonc_parser.parser import JsoncParser
except Exception:
    JsoncParser = None

from kt_unlearn.utils.config.composer import compose


def _strip_jsonc(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = re.sub(r"/\*[\s\S]*?\*/", "", text, flags=re.MULTILINE)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"(^|\s)//.*$", "", line)
        lines.append(line)
    return "\n".join(lines)

class Config:
    def __init__(self, file):
        self.__dict__ = file

    @classmethod
    def from_json(cls, json_file):
        if JsoncParser is not None:
            return cls(compose(JsoncParser.parse_file(json_file)))
        with open(json_file, "r", encoding="utf-8-sig") as f:
            raw = json.loads(_strip_jsonc(f.read()))
        return cls(compose(raw))

        
    
    
