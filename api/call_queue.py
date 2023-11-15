import os
import queue
import pathlib


class LocalFileQueue:

    file_name = 'api_queue.txt'

    def __init__(self, file_path, file_name=file_name):
        if os.path.exists(file_path) is False:
            pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue()
        self.file_path = f'{file_path}/{file_name}'
        self.restore_from_disk()

    def enqueue(self, data):
        self.queue.put(data)
        self.persist_to_disk(data)

    def dequeue(self, is_complete=True):
        data = self.queue.get()
        if is_complete:
            self.complete(data)
        return data

    def is_empty(self):
        return self.queue.empty()

    def size(self):
        return self.queue.qsize()

    def persist_to_disk(self, data):
        with open(self.file_path, 'a') as file:
            file.write(data + '\n')

    def complete(self, data):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        with open(self.file_path, 'w') as file:
            for line in lines:
                if line.strip() != data:
                    file.write(line)

    def restore_from_disk(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                self.queue.put(line.strip())

    def insert_at_front(self, data):
        temp_queue = queue.Queue()
        temp_queue.put(data)
        while not self.queue.empty():
            temp_queue.put(self.queue.get())
        self.queue = temp_queue
        
