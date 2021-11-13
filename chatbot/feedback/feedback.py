import random

def intent_feedback(ner_result):
    """
    generate the feedback query when ner is detected without intent.
    Args:
        ner_result list[str]: the ner result of client query.
    """
    reply_list = []

    for ner in ner_result:
        intent, value = ner.split('\t')
        if intent == "景点-名称":
            reply_list.append(f"{value}附近有什么酒店？")
            reply_list.append(f"{value}在哪里？")
            reply_list.append(f"{value}周围有什么好吃的？")

        elif intent == "地铁-出发地":
            reply_list.append(f"{value}在哪里？")

        elif intent == "餐馆-名称":
            reply_list.append(f"{value}有什么推荐菜？")
            reply_list.append(f"{value}在哪里？")
            reply_list.append(f"{value}评分怎么样？")


        elif intent == "餐馆-推荐菜":
            reply_list.append(f"帮我找一家有{value}的餐馆。")

        elif intent == "餐馆-营业时间":
            reply_list.append(f"{value}的时候，故宫附近有什么餐馆在营业？")

        elif intent == "景点-门票":
            reply_list.append(f"帮我找个免费的景点。")
            reply_list.append(f"帮我找个我买不起门票的景点。")

        elif intent == "景点-评分":
            reply_list.append(f"帮我找个评分为{value}的景点。")

    reply = "抱歉，小七没听懂。您是不是想问："

    if len(reply_list) > 4:
        random.shuffle(reply_list)

    count = 0
    for sentence in reply_list:
        reply += f"<br>{sentence}"
        count += 1
        if count >= 4:
            break

    return reply, len(reply_list)