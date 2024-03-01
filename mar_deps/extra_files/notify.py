import os

import requests


class NotifierClient:
    def __init__(self):
        self.notify_host = os.environ.get("NOTIFY_HOST", "http://192.168.30.136:19087")

    def success(self, name, task_id, path):
        resp = requests.post(f"{self.notify_host}/gds/inference-result", json={
            "name": name,
            "taskId": task_id,
            "path": path
        })
        print(resp.json())

    def failure(self, task_id):
        resp = requests.put(f"{self.notify_host}/gds/inference-task", data={
            "id": task_id,
            "state": -1
        })
        print(resp.json())
