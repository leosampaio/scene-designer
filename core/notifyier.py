import slack_sdk
import sys
import threading
import os.path
import pickle
from functools import wraps

default_channel = 'sketch-transformer'


class Notifyier(object):

    def __init__(self, config_file=None, save_dir=None):
        self.dummy = config_file is None or config_file == ''
        if not self.dummy:
            with open(config_file) as f:
                self.token = f.readline().strip()
                self.channel = f.readline().strip()
        if not self.dummy and save_dir is not None:
            self.filename = os.path.join(save_dir, "notifyier.pickle")
        else:
            self.filename = None
        if self.filename is not None and os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                self.slack_threads = pickle.load(f)
        else:
            self.slack_threads = {}

    def send_if_not_dummy(send):
        @wraps(send)
        def wrapper(inst, *args, **kwargs):
            if inst.dummy:
                print("[Notification] Not sent because there is no token")
                return
            else:
                send(inst, *args, **kwargs)
        return wrapper

    def _send_initial_message(self, sc, experiment_id):
        """Starts the thread on slack by sending the initial message with the
        command line arguments and experiment ID
        """
        message = "*[{}]*\n`python {}`".format(experiment_id, ' '.join(sys.argv))
        response = sc.chat_postMessage(
            channel=self.channel,
            text=message,
            as_user=True
        )
        self.slack_threads[self.channel] = response["ts"]
        if self.filename is not None:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.slack_threads, f)

    @send_if_not_dummy
    def notify_with_message(self, message, experiment_id, send_to_channel=False):
        """Sends a message to the thread associated with this experiment_id
        """
        try:
            sc = slack_sdk.WebClient(self.token)
            if self.slack_threads.get(self.channel) is None:
                self._send_initial_message(sc, experiment_id)
            sc.chat_postMessage(
                channel=self.channel,
                text=message,
                thread_ts=self.slack_threads[self.channel],
                reply_broadcast=send_to_channel,
                as_user=True
            )
        except Exception as e:
            print(repr(e))

    def _notify_with_image(self, imagepath, experiment_id, message):
        try:
            sc = slack_sdk.WebClient(self.token)
            sc.files_upload(title=message,
                            channels=self.channel,
                            thread_ts=self.slack_threads[self.channel],
                            file=imagepath)
        except Exception as e:
            print(repr(e))

    @send_if_not_dummy
    def notify_with_image(self, imagepath, experiment_id, message=None):
        """Sends an image to the thread associated with this experiment_id
        """
        try:
            if message is None:
                message = imagepath
            if self.slack_threads.get(self.channel) is None:
                sc = slack_sdk.WebClient(self.token)
                self._send_initial_message(sc, experiment_id)
            os_thread = threading.Thread(target=self._notify_with_image,
                                         args=(imagepath, experiment_id, message))
            os_thread.start()
        except Exception as e:
            print(repr(e))
