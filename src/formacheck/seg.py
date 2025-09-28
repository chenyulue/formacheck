import hanlp

tok = hanlp.load("COARSE_ELECTRA_SMALL_ZH")
pos = hanlp.load("CTB9_POS_ELECTRA_SMALL")
tok.config.output_spans = True
tok.dict_combine = {"所述的", "所述", "上述", "上述的", "前述", "前述的", "该"}

_NOUN_TAGS = set("NN NR NT CD OD M JJ DT PN".split())
_STOP_TAGS = set("w c p u y e o g x".split())


def get_noun_phrases(claim: str) -> dict[str, list[int]]:
    """
    获取权利要求中“所述”之后的名词性短语, 及其在权利要求中的起始字符的位置.
    注意: 传入的权利要求claim需要预先去除空格, 以保证分词准确性.

    Args:
        sentence (str): 待分词的权利要求.

    Returns:
        dict[str, list[int]]: 键为获取的名词性短语，值为该短语在权利要求中的起始位置索引列表.
    """
    words = tok(claim)
    tags = pos([word[0] for word in words])
    noun_phrases: dict[str, list[int]] = {}
    i = 0
    while i < len(words):
        word, start, _end = words[i]
        if word in tok.dict_combine:
            i += 1
            buf = [words[i][0]]
            current_pos = words[i][1]
            i += 1
            while i < len(words) and tags[i] in _NOUN_TAGS:
                buf.append(words[i][0])
                i += 1
            noun_phrase = "".join(buf)
            noun_phrases.setdefault(noun_phrase, []).append(current_pos)
            continue
        i += 1

    return noun_phrases


if __name__ == "__main__":
    claim = """8.一种半导体结构的形成方法，其特征在于，包括：
提供基底，所述基底上形成有栅极结构，所述栅极结构两侧的基底中形成有源漏掺杂层，所述栅极结构侧部的基底上形成有覆盖所述源漏掺杂层的源漏互连层，且所述源漏互连层与源漏掺杂层电连接；
沿所述基底顶面的法线方向，去除部分厚度的所述源漏互连层，形成位于相邻所述栅极结构之间的凹槽；
在所述凹槽底部形成覆盖所述源漏互连层的保护层；
形成所述保护层后，对所述源漏互连层表面进行还原处理，用于实现去氧化；
在所述还原处理后，在所述凹槽中填充源漏盖帽层。"""
    words = tok(claim)
    tags = pos([word[0] for word in words])
    for w, t in zip(words, tags):
        print(w[0], "=>", t)
    print("Result:", get_noun_phrases(claim))
