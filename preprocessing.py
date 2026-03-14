def preprocess_data(df):
    df = df[["Date", "Close"]].copy()

    df.dropna(inplace=True)
    df.loc[:, "MA_7"] = df["Close"].rolling(7).mean()
    df.loc[:, "MA_30"] = df["Close"].rolling(30).mean()
    df.loc[:, "Returns"] = df["Close"].pct_change()

    return df
