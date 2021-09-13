import pandas as pd
import numpy as np

s1 = pd.Series([1, 2, 3, 4, 5, 6])
s2 = pd.Series(["a", "b", "c", "d", "e", "f"])
df = pd.DataFrame({"numbers": s1, "alphabet": s2})
#print(df)
#print(df["numbers"].isin([2, 5]))
df = df[df["numbers"].isin([2, 5])]
print(df)
