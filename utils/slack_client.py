#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:34:32 2018
slack_client.py
send a message to my self
This is a bot named 'Bot Tu' in workspace "Digital Creativity Lab"
Note: to get list of users slack_client.api_call('im.list')
to get list of channels: slack_client.api_call('conversations.list')

@author: Tu Bui tb00083@surrey.ac.uk
"""
import sys
import os
import numpy as np
try:  # slack v1.x
    from slackclient import SlackClient
    SLACK_VERSION = 1
except:  # slack v2.x
    import slack
    SLACK_VERSION = 2

TU_TOKEN = 'xoxb-411990984932-532547748294-5cRLcxZjZzP9OkeVb0dyURu6'  # bot tu token


class Slack(object):
    def __init__(self, channel=None, token=None):
        if token is None:
            self.token = TU_TOKEN
        else:
            self.token = token
        self.myID = 'DFLBLKQ6L'  # send to myself
        self.channel = []
        self.set_channel(channel)
        if SLACK_VERSION == 1:
            self.slack_client = SlackClient(self.token)
            assert self.slack_client.rtm_connect(with_team_state=False), "Error. Cannot connect to slack"
        elif SLACK_VERSION == 2:
            self.slack_client = slack.WebClient(token=self.token)

    def set_channel(self, channel):
        """setup channel(s) to send msg/file to"""
        if channel is None:
            self.channel = [self.myID, ]
        elif isinstance(channel, str):
            self.channel = [channel, ]
        elif isinstance(channel, list):
            self.channel = channel
        else:
            sys.exit('Channel must be string or list of string. Quiting Slack.')

    def send_msg(self, text='hi'):
        """send a message to user/channel/group
        return True if msg send successfully to at least 1 channel"""
        out = []
        for channel in self.channel:
            if SLACK_VERSION == 1:
                res = self.slack_client.api_call('chat.postMessage', channel=channel, text=text)
            elif SLACK_VERSION == 2:
                res = self.slack_client.chat_postMessage(channel=channel, text=text)
            out.append(res['ok'])
        return np.any(out)

    def send_file(self, file_path):
        """send a file to user/channel/group"""
        if os.path.isfile(file_path):
            if SLACK_VERSION ==1 :
                res = self.slack_client.api_call('files.upload', channels=self.channel,
                                                 file=open(file_path, 'rb'))
            elif SLACK_VERSION == 2:
                for channel in self.channel:
                    res = self.slack_client.files_upload(channels=channel, file=file_path)
            return res['ok']
        else:
            return False
