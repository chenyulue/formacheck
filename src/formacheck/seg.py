from typing import Any
import itertools
import hanlp
from hanlp_common.document import Document

user_dict = {"所述的", "所述", "上述", "上述的", "前述", "前述的", "该"}

tok = hanlp.load("COARSE_ELECTRA_SMALL_ZH")
pos = hanlp.load("CTB9_POS_ELECTRA_SMALL")
con = hanlp.load("CTB9_CON_FULL_TAG_ELECTRA_SMALL")
tok.config.output_spans = True
tok.dict_combine = user_dict


def extract_span_start(tokspans: list[Any]) -> list[int]:
    """从三元组[单词, 单词起始下标, 单词结束下标]中提取单词起始下标

    Parameters
    ----------
    tokspans : list[Any]
        每一个单词的三元组为[单词, 单词起始下标, 单词结束下标]

    Returns
    -------
    list[int]
        每个单词的起始下标
    """
    return [t[1] for t in tokspans]


def extract_tokens(tokspans: list[Any]) -> list[str]:
    """从三元组[单词, 单词起始下标, 单词结束下标]中提取单词

    Parameters
    ----------
    tokspans : list[Any]
        每一个token的三元组为[单词, 单词起始下标, 单词结束下标]

    Returns
    -------
    list[str]
        单词列表
    """
    return [t[0] for t in tokspans]


def merge_pos_start_into_con(doc: Document) -> Document:
    """将pos标签和单词的起始下标start合并到con中

    Parameters
    ----------
    doc : Document
        包含con, pos, start的Document

    Returns
    -------
    Document
        合并后的Document
    """
    # 该函数逻辑完全照搬http://localhost:8888/lab/tree/Untitled.ipynb中的代码,
    # 只是添加想单词的起始下标start集成
    flat = isinstance(doc["pos"][0], str)
    if flat:
        doc = Document((k, [v]) for k, v in doc.items())
    for tree, tags, starts in zip(doc["con"], doc["pos"], doc["start"]):
        offset = 0
        for subtree in tree.subtrees(lambda t: t.height() == 2):
            tag = subtree.label()
            if tag == "_":
                subtree.set_label(f"{starts[offset]}-{tags[offset]}")
            offset += 1
    if flat:
        doc = doc.squeeze()
    return doc


_nlp = (
    hanlp.pipeline()
    .append(tok, output_key="toks")
    .append(extract_tokens, input_key="toks", output_key="tok")
    .append(extract_span_start, input_key="toks", output_key="start")
    .append(pos, input_key="tok", output_key="pos")
    .append(con, input_key="tok", output_key="con")
    .append(merge_pos_start_into_con, input_key="*")
)


def extract_technical_feature(claim: str) -> dict[str, list[int]]:
    """
    获取权利要求中“所述”之后的名词性短语, 及其在权利要求中的起始字符的位置.

    Args:
        sentence (str): 待分词的权利要求.

    Returns:
        dict[str, list[int]]: 键为获取的名词性短语，值为该短语在权利要求中的起始位置索引列表.
    """
    doc = _nlp(claim)

    nps = {}
    for subtree in doc["con"].subtrees(lambda t: t.label().startswith("NP")):
        leaves = set(subtree.leaves())
        if not any(map(lambda g: g in leaves, user_dict)):
            continue
        
        leaves = [leaf for leaf in itertools.dropwhile(lambda x: x[0] not in tok.dict_combine, subtree.pos())]
        if len(leaves) > 1:
            nps.setdefault(leaves[0][1].split('-')[0], []).append(leaves)

    nps_select = {k: min(v, key=lambda x: len(x)) for k, v in nps.items()}

    tech_features = {}
    for v in nps_select.values():
        pairs = []
        for pair in v:
            if pair[1].split('-')[1] == "PU" or pair[0] == "的":
                break
            if pair[0] not in tok.dict_combine:
                pairs.append(pair)
        # pairs = [pair for pair in v if pair[0] not in tok.dict_combine]
        tech_features.setdefault(''.join(w[0] for w in pairs), list()).append(int(pairs[0][1].split('-')[0]))

    return tech_features


if __name__ == "__main__":
    claim = """1.一种GaN基HEMT器件结构的制作方法，其特征在于，包括以下步骤： 提供一基底，所述基底中设有GaN-AlGaN异质结层； 形成栅极图形层于所述基底上，所述栅极图形层包括栅极环阵列与栅极线组，所述栅极环阵列包括呈六角阵列排布的多个六边形栅极环，所述栅极线组包括沿第一方向延伸的栅极主线及多条平行排列且均往第二方向延伸的栅极子线，所述第一方向与所述第二方向均平行于所述基底所在平面，且所述第一方向与所述第二方向相互交叉，所述栅极环阵列与多条所述栅极子线位于所述栅极主线的同一侧，多个所述六边形栅极环划分为多个平行排列的栅极环组，每个所述栅极环组包括沿所述第二方向依次且间隔排列的多个所述六边形栅极环，多条所述栅极子线与多个所述栅极环组一一对应连接，其中，每条所述栅极子线包括间断设置的多条栅极子线单元，所述栅极环组中的任意相邻两个所述六边形栅极环之间通过一所述栅极子线单元连接，所述栅极环组中最靠近所述栅极主线的一个所述六边形栅极环通过一所述栅极子线单元连接至所述栅极主线； 形成覆盖所述栅极图形层的第一介质层； 形成源漏极图形层于所述第一介质层上，所述源漏极图形层包括呈六角阵列排布的多个六边形电极板及呈六角阵列排布的多个六边形电极环，所述六边形电极板与所述六边形电极环中的一个作为源极，另一个作为漏极，每一所述六边形电极板在所述栅极图形层所在平面上的垂直投影对应设置于一所述六边形栅极环的中心，每一所述六边形电极环在所述栅极图形层所在平面上的垂直投影对应围设于一所述六边形栅极环的四周，任意相邻两个所述六边形电极环共用一侧边，所述六边形电极环在所述栅极图形层所在平面上的垂直投影在所述栅极子线单元的经过路径上具有缺口。"""
    # words = tok(claim)
    # tags = pos([word[0] for word in words])
    # for w, t in zip(words, tags):
    #     print(w[0], "=>", t)
    print(extract_technical_feature(claim))
