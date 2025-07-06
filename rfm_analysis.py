import pandas as pd
from sklearn.cluster import KMeans

def calculate_rfm(df, customer_id_col, date_col, amount_col):
    df[date_col] = pd.to_datetime(df[date_col])
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        customer_id_col: 'count',
        amount_col: 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def perform_kmeans_clustering(rfm, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = model.fit_predict(rfm)
    return rfm
