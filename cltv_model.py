from lifetimes import BetaGeoFitter, GammaGammaFitter
import pandas as pd

def fit_bg_nbd(df, frequency_col, recency_col, T_col):
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(df[frequency_col], df[recency_col], df[T_col])
    return bgf

def fit_gamma_gamma(df, frequency_col, monetary_col):
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(df[frequency_col], df[monetary_col])
    return ggf

def calculate_cltv(bgf, ggf, df, freq='W', time=12, discount_rate=0.01):
    cltv = ggf.customer_lifetime_value(
        bgf,
        df['frequency'],
        df['recency'],
        df['T'],
        df['monetary_value'],
        time=time,
        freq=freq,
        discount_rate=discount_rate
    )
    return cltv
