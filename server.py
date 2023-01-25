#!/usr/bin/env python

import argparse
import connexion
import os
import yaml
from flask import send_from_directory, redirect
from flask_cors import CORS
import docx2txt

from backend import AVAILABLE_MODELS

#CONFIG_FILE_NAME = 'lmf.yml'
projects = {}

app = connexion.App(__name__, debug=False)


class Project:
    def __init__(self, LM, config):
        self.config = config
        self.lm = LM()


def get_all_projects():
    res = {}
    for k in projects.keys():
        res[k] = projects[k].config
    return res


def analyze(analyze_request):
    project = analyze_request.get('project')
    path = "C:/Users/Pavani Rangineni/Downloads/test.docx"
    text = docx2txt.process(path)

    res = {}
    if project in projects:
        p = projects[project] # type: Project
        res = p.lm.check_probabilities(text, topk=20)

    return {
        "request": {'project': project, 'text': text},
        "result": res
    }


app.add_api('server.yaml')

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='gpt-2-small')
parser.add_argument("--nodebug", default=True)
parser.add_argument("--address",
                    default="127.0.0.0")  # 0.0.0.0 for nonlocal use
parser.add_argument("--port", default="5001")
parser.add_argument("--nocache", default=False)
parser.add_argument("--dir", type=str, default=os.path.abspath('data'))

parser.add_argument("--no_cors", action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.no_cors:
        CORS(app.app, headers='Content-Type')

    app.run(port=int(args.port), debug=not args.nodebug, host=args.address)
else:
    args, _ = parser.parse_known_args()
    try:
        model = AVAILABLE_MODELS[args.model]
    except KeyError:
        print("Model {} not found. Make sure to register it.".format(
            args.model))
        print("Loading GPT-2 instead.")
        model = AVAILABLE_MODELS['gpt-2']
    projects[args.model] = Project(model, args.model)
