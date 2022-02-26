from pathlib import Path

import pandas as pd
import pqr


def read_prices(path: str) -> pd.DataFrame:
    path = Path(path)
    prices = []

    for filepath in path.iterdir():
        current_prices = pd.read_csv(
            filepath,
            index_col="Date",
            parse_dates=True
        ).loc[:, "Close"].rename(
            filepath.parts[-1].replace(".csv", "")
        )

        prices.append(current_prices)

    prices = pqr.utils.align(*prices)

    return pd.DataFrame(prices).T


def read_factor(
        factor: str,
        path: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    factor_data = pd.pivot_table(
        df,
        index="calendardate",
        columns="ticker",
        values=factor,
    )
    factor_data.index = pd.to_datetime(factor_data.index)

    return factor_data.resample("D").asfreq().ffill()
