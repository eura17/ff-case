from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pqr


def vix_scaling(
        holdings: pd.DataFrame,
        vix: pd.Series,
        target: float,
        n: int,
) -> pd.DataFrame:
    mean_vix = vix.rolling(n).mean().shift().iloc[n:]
    holdings, mean_vix = pqr.utils.align(
        holdings,
        mean_vix,
    )
    mean_vix.name = holdings.columns[0]

    return holdings * pd.DataFrame(target / mean_vix,
                                   columns=holdings.columns)


def grid_search(
        targets: list[float],
        ns: list[int],
        volatility: Callable[[pd.DataFrame], pd.DataFrame],
        df: pd.DataFrame,
        metric: Callable[[pqr.Portfolio], float],
) -> pd.DataFrame:
    grid = pd.DataFrame(
        np.nan,
        index=pd.Index(ns, name="n"),
        columns=pd.Index(targets, name="target"),
    )

    for n in ns:
        for target in targets:
            portfolio = pqr.Portfolio.backtest(
                longs=df[["VFINX"]].astype(bool),
                shorts=None,
                allocator=pqr.utils.compose(
                    lambda holdings: holdings.astype(float),
                    pqr.utils.partial(
                        volatility,
                        n=n,
                        target=target,
                    ),
                    pqr.utils.partial(
                        pqr.limit_leverage,
                        min_leverage=0.3,
                        max_leverage=1,
                    ),
                ),
                calculator=pqr.utils.partial(
                    pqr.calculate_returns,
                    universe_returns=pqr.prices_to_returns(df[["VFINX"]]),
                ),
            )

            grid.loc[n, target] = metric(portfolio)

    return grid
