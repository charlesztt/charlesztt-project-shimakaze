import requests
import json


class PassiveBot:
    def __init__(self, config_json_path="shimakaze.json"):
        self.__json_config = json.load(open(config_json_path))
        self.chat_id = self.__json_config["telegram_bot"]["chat_id"]
        self.bot_token = self.__json_config["telegram_bot"]["bot_token"]
        self.headers = {'content-type': 'application/json'}
        self.url_base = 'https://api.telegram.org/bot%s' % self.bot_token

    def send_message(self, input_message):
        url = '%s/sendMessage' % self.url_base
        object_dict = dict()
        object_dict["chat_id"] = self.chat_id
        object_dict["text"] = input_message
        response = requests.post(url, data=json.dumps(object_dict), headers=self.headers)
        if response.status_code == 200:
            print("Message Sent")
        else:
            raise Exception("Error!")
