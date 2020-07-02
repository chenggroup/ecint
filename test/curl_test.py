import json

import requests


def notification_in_dingtalk(webhook, node):
    headers = {'Content-Type': 'application/json'}
    title = 'Job Info'
    text = '## Job Info\n'
    text += 'Your job is over!\n'
    text += '>\n'
    text += f'> Job PK: **{node.pk}**\n'
    text += '>\n'
    text += f'> Job Chemical Formula: **{node.inputs.structure.get_formula()}**\n'
    text += '>\n'
    text += f'> Job Type: **{node.process_label}**\n'
    text += '>\n'
    text += f'> Job State: **{node.process_state.name}**\n'
    data = {'msgtype': 'markdown', 'markdown': {'title': title, 'text': text}}
    response = requests.post(url=webhook, headers=headers, data=json.dumps(data))
    return response


if __name__ == '__main__':
    from aiida.orm import load_node
    from aiida import load_profile

    load_profile()
    webhook = 'https://oapi.dingtalk.com/robot/send?access_token=a3cd7e35c31f149248a46053f51b11ad843cc50a975730e565cb3f0292f8e56b'
    node = load_node(1144)
    notification_in_dingtalk(webhook, node)
