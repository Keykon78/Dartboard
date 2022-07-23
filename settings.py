CAMERA = {
    "INPUT": 2
}
RUNTIME = {
    "MODE": 'debug'
}


class Settings():

    @staticmethod
    def get_cam_input():
        return CAMERA["INPUT"]

    @staticmethod
    def get_run_mode():
        return RUNTIME["MODE"]
