# -*- coding: utf-8 -*-

__all__ = ["CustomLogger"]

import io


class CustomLogger:
    def __init__(self, stream: io.IOBase, file: io.IOBase):
        self.stream = stream
        self.file = file

    def write(self, text: str):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
