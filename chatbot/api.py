from flask import Flask
from flask.globals import request
from flask.json import jsonify
from flask_cors import CORS
# from open_chat.open_chat_util import OpenChat
# 可以添加闲聊模块

import config
import sys

sys.path.append("intent")
sys.path.append("bislot")
sys.path.append("ner")

from intent.predict import IntentPrediction
from bislot.predict import BiPrediction
from ner.predict import NerPredicter
from feedback import feedback
from database_code.database_main import DataBase
from response_code.response_main import rule_response
from entitylinking import linking_utils
from collections import defaultdict

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route("/sendmsg", methods=['POST'])
def sendMsg2Robot():
    req = request.form.get('content')
    data1, data2, data3, data4, resp = FitAndPredict(req)
    return jsonify({
        "data": {
            "data1": data1,
            "data2": data2,
            "data3": data3,
            "data4": data4,
            "resp": resp
        }
    })


args = config.get_args()
print("loading intent")
intent_pre = IntentPrediction(args, intent_weight=config.intent_weight)
print("loading bi")
bi_pre = BiPrediction(args,
                      bi_weight=config.bi_weight,
                      id2bi_label=config.id2bi_label,
                      bi_label2id=config.bi_label2id
                      )
print("loading ner")

ner_pre = NerPredicter(model_path=config.ner_weight,
                       ner_id2label=config.ner_id2label,
                       args=args,
                       ner_label2id=config.ner_label2id)
# print("loading open-field chat")
# open_chat = OpenChat()

database = DataBase()
DST = defaultdict(set)
mapping = linking_utils.get_link_map("entitylinking/data/place.txt")


def FitAndPredict(content):
    global DST
    print("===" * 15)
    print("用户query:\t", content)
    if content == 'clear':
        DST = defaultdict(set)
        return '', '', '', '', '你点了什么?!'
    elif content == "":
        return '', '', '', '', '?'
    else:
        intents = intent_pre.intent_predict(content)
        bi_entities = bi_pre.bi_predict(content)
        ner_entities = ner_pre.ner_predict(content)

        if len(intents) == 0:
            if len(ner_entities) > 0:
                # reply intent guess
                reply, len_reply = feedback.intent_feedback(ner_result=ner_entities)
                if len_reply > 0:
                    return '', '', '', '', reply
            else:
                pass
                # 闲聊添加
                # answer = open_chat.predict(content)
                # return '', '', '', '', answer

        if len(bi_entities) != 0:
            DST['酒店-酒店设施'] = [bi_entity.split('-')[-1] for bi_entity in bi_entities]

        print("\nNER 结果：")
        for entity in ner_entities:
            domain_slot, value = entity.split('\t')
            print(f"检测到实体 {value}, 类别 {domain_slot}", end="\t")
            official_name = linking_utils.link_entity(value, mapping)
            if official_name is not None:
                print(f"实体被链接到 {official_name}")
                value = official_name
            DST[domain_slot] = [value]
            print()

        print("\nDST（当前状态）:")
        for item in DST:
            print(f"{item}:\t{DST[item]}")

        print("\n（意图识别结果）用户询问：\t", " ".join(intents))

        answer, DST = rule_response(content, intents, database, DST)

        str_DST = ''
        for key, value in DST.items():
            if isinstance(value, str):
                str_DST += key + '：' + value + '\n'
            else:
                str_DST += key + '：' + ','.join(value) + '\n'

        return None, None, None, str_DST, answer


if __name__ == '__main__':
    webhost = "0.0.0.0"
    webport = 5000
    app.run(host=webhost, port=webport)
