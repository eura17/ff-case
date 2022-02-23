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
