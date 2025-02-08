# Numpy and Pandas practices

* Use `np.isnan(val)` rather than `val == np.nan` to perform comparison

## Parquet rather than csv

Apache *Parquet* is an open source, column-oriented data file format designed for efficient data storage and retrieval. It provides efficient data compression and encoding schemes with enhanced performance to handle complex data in bulk.

```py
import pyarrow 
import pandas as pd
#read parquet file into pandas dataframe
df=pd.read_parquet('file_location/file_path.parquet',engine='pyarrow')
#writing dataframe back to source file
df.to_parquet('file_location/file_path.parquet', engine='pyarrow')
```