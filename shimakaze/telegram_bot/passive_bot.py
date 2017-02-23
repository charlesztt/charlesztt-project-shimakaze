import requests
import json

from shimakaze.util.cfg_reader import CfgReader

class PassiveBot:
    def __init__(self):
        cr = CfgReader("shimakaze.cfg")
        self.chat_id = cr.get_key_from_session("telegram_bot", "chat_id")
        self.bot_token = cr.get_key_from_session("telegram_bot", "bot_token")
        self.headers = {'content-type': 'application/json'}
        self.url_base = 'https://api.telegram.org/bot%s'%self.bot_token

    def send_message(self, input_message):
        url = '%s/sendMessage'%self.url_base
        object_dict = dict()
        object_dict["chat_id"] = self.chat_id
        object_dict["text"] = input_message
        response = requests.post(url, data=json.dumps(object_dict), headers=self.headers)
        if response.status_code == 200:
            print("Message Sent")
        else:
            raise Exception("Error!")