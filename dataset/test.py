import pandas as pd

data = []
for row in some_function_that_yields_data():
    data.append(row)

pd.DataFrame(data)