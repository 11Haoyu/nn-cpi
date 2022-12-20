import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from deep_cpi.data.meta import get_all_dataset_metas

INDICATORS1 = [
    # 1948-01   monthly
    "美国:失业率:季调",
    # 1913-05   monthly
    "美国:CPI:环比",
    # 1914-01   monthly
    "美国:CPI:当月同比",
    # 1959-01   monthly
    "美国:M1",
    # 1959-01   monthly
    "美国:M2",
    # 1952-11   quarterly
    "美国:密歇根大学消费者预期指数",
    # 1970-12   yearly
    "OECD:产出缺口:美国",
    # 1986-01   daily
    "LIBOR:美元:3个月",
    # 1959-01   monthly
    "美国:个人可支配收入(不变价):季调:折年数",
]
INDICATORS2 = [
    # 2006-01   daily
    "名义美元指数:广义",
    # 1992-01   monthly
    "美国:贸易差额:季调",
    # 1972-01   monthly
    "美国:工业总产值:最终产品:季调",
    # 1992-01   monthly
    "美国:零售和食品服务销售额:总计",
    # 1998-10   monthly
    "美国:外汇",
    # 1963-01   monthly
    "美国:新建住房销售:折年数:季调",
    # 1971-02   daily
    "美国:纳斯达克综合指数",
    # 1997-12   yearly
    "工业增加值:按本币计算:不变价:美国",
    # 1947-03   quarterly
    "美国:联邦政府:经常性支出:总计:未季调",
    # 2003-06   quarterly
    "美国:外债总额",
    # 1998-08   weekly
    "汽油价格:常规零售:美国",
]

EXTRA_INDICATORS = ["日期", "美国:CPI:环比", "美国:CPI:当月同比"]


def main():
    df1 = pd.read_excel("data/data_raw1.xlsx")
    df2 = pd.read_excel("data/data_raw2.xlsx")
    df = pd.merge(df1, df2, on="日期", how="outer")

    metas = get_all_dataset_metas()

    for name, meta in metas.items():
        date_begin, date_end = meta.date_range
        indicators = meta.indicators

        sub_df = df[(df["日期"] >= date_begin) & (df["日期"] <= date_end)]
        indicators = ["日期", "美国:CPI:环比", "美国:CPI:当月同比"] + indicators
        sub_df = sub_df[indicators]
        sub_df = sub_df.set_index("日期").resample("M").interpolate(
            limit_direction="both"
        )
        sub_df = sub_df.dropna()

        save_path = f"data/{name}_raw.csv"
        sub_df.to_csv(save_path)
        print(f"{save_path} saved.")

        columns_to_scale = sub_df.columns.drop(["美国:CPI:环比", "美国:CPI:当月同比"])
        sub_df[columns_to_scale] = sub_df[columns_to_scale].apply(
            lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).reshape(-1)
        )

        save_path = f"data/{name}_scaled.csv"
        sub_df.to_csv(save_path)
        print(f"{save_path} saved.")


if __name__ == "__main__":
    main()
