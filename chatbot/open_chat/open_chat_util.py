from paddlenlp.transformers import UnifiedTransformerLMHeadModel
from paddlenlp.transformers import UnifiedTransformerTokenizer


def post_process_response(token_ids, tokenizer):
    """
    Post-process the decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
    """
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    response = tokenizer.merge_subword(tokens)
    return token_ids, response


def get_in_turn_repetition(pred, is_cn=True):
    """Get in-turn repetition."""
    if len(pred) == 0:
        return 1.0
    if isinstance(pred[0], str):
        pred = [tok.lower() for tok in pred]
        if is_cn:
            pred = "".join(pred)
    tri_grams = set()
    for i in range(len(pred) - 2):
        tri_gram = tuple(pred[i:i + 3])
        if tri_gram in tri_grams:
            return True
        tri_grams.add(tri_gram)
    return False


def select_response(ids, scores, tokenizer, max_dec_len=None, num_samples=1):
    ids = ids.numpy().tolist()
    scores = scores.numpy()

    if len(ids) != len(scores) or (len(ids) % num_samples) != 0:
        raise ValueError(
            "the length of `ids` is {}, but the `num_samples` is {}".format(
                len(ids), num_samples))

    group = []
    tmp = []
    for pred, score in zip(ids, scores):
        pred_token_ids, pred_tokens = post_process_response(pred, tokenizer)
        num_token = len(pred_token_ids)
        response = "".join(pred_tokens)

        in_turn_repetition = get_in_turn_repetition(
            pred_tokens, True) or get_in_turn_repetition(pred_token_ids)
        if max_dec_len is not None and num_token >= max_dec_len:
            score -= 1e3
        elif in_turn_repetition:
            score -= 1e3

        tmp.append([response, score])
        if len(tmp) == num_samples:
            group.append(tmp)
            tmp = []

    results = []
    for preds in group:
        preds = sorted(preds, key=lambda x: -x[1])
        results.append(preds[0][0])
    return results


class OpenChat():
    def __init__(self):
        self.pretrain_model_name = 'unified_transformer-12L-cn'  #
        # self.pretrain_model_name = 'plato-mini'
        self.model = UnifiedTransformerLMHeadModel.from_pretrained(self.pretrain_model_name)
        self.tokenizer = UnifiedTransformerTokenizer.from_pretrained(self.pretrain_model_name)

    def predict(self, content):
        inputs = self.tokenizer.dialogue_encode(
            content,
            return_tensors=True,
            is_split_into_words=False)
        ids, scores = self.model.generate(**inputs,
                                          decode_strategy='beam_search',
                                          num_beams=6,
                                          max_dec_len=20,
                                          min_dec_len=1,
                                          repetition_penalty=1.2,
                                          length_penalty=0.3
                                          )
        # print(ids)
        results = select_response(ids, scores, self.tokenizer, max_dec_len=20)[0]
        return results


if __name__ == '__main__':
    content = "今天天气怎么样"
    open_chat = OpenChat()
    result = open_chat.predict(content)
    print(result)