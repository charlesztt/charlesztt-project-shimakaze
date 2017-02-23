import configparser


class CfgReader:
    def __init__(self, cfg_file_path):
        self.config = configparser.ConfigParser()
        self.config.read(cfg_file_path)

    def get_key_from_session(self, section, option):
        return self.config.get(section, option)