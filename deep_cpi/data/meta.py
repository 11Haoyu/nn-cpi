from dataclasses import dataclass


@dataclass
class DatasetMeta:
    name: str
    date_range: tuple[str, str]
    indicators: list[str]


DATASET_METAS = {
    "CPIv1": DatasetMeta(
        name="CPIv1",
        date_range=("1959-01-01", "2022-07-31"),
        indicators=[
            # group #1
            "美国:失业率:季调",
            "美国:M1",
            "美国:M2",
            "美国:密歇根大学消费者预期指数",
            "美国:个人可支配收入(不变价):季调:折年数",
            "美国:联邦政府:经常性支出:总计:未季调",
        ],
    ),
    "CPIv2": DatasetMeta(
        name="CPIv2",
        date_range=("1963-01", "2022-07-31"),
        indicators=[
            # group #1
            "美国:失业率:季调",
            "美国:M1",
            "美国:M2",
            "美国:密歇根大学消费者预期指数",
            "美国:个人可支配收入(不变价):季调:折年数",
            "美国:联邦政府:经常性支出:总计:未季调",
            # group #2
            "美国:新建住房销售:折年数:季调",
        ],
    ),
    "CPIv3": DatasetMeta(
        name="CPIv3",
        date_range=("1972-01", "2022-07-31"),
        indicators=[
            # group #1
            "美国:失业率:季调",
            "美国:M1",
            "美国:M2",
            "美国:密歇根大学消费者预期指数",
            "美国:个人可支配收入(不变价):季调:折年数",
            "美国:联邦政府:经常性支出:总计:未季调",
            # group #2
            "美国:新建住房销售:折年数:季调",
            # group #3
            "OECD:产出缺口:美国",
            "美国:工业总产值:最终产品:季调",
            "美国:纳斯达克综合指数",
        ],
    ),
    "CPIv4": DatasetMeta(
        name="CPIv4",
        date_range=("1986-01", "2022-07-31"),
        indicators=[
            # group #1
            "美国:失业率:季调",
            "美国:M1",
            "美国:M2",
            "美国:密歇根大学消费者预期指数",
            "美国:个人可支配收入(不变价):季调:折年数",
            "美国:联邦政府:经常性支出:总计:未季调",
            # group #2
            "美国:新建住房销售:折年数:季调",
            # group #3
            "OECD:产出缺口:美国",
            "美国:工业总产值:最终产品:季调",
            "美国:纳斯达克综合指数",
            # group #4
            "LIBOR:美元:3个月",
        ],
    ),
}


def get_all_dataset_metas() -> dict[str, DatasetMeta]:
    return DATASET_METAS
