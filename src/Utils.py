import pandas as pd

def display_dataframe_info(df: pd.DataFrame, msg: str = 'Displaying the df metadata'):


    print('#' * 100)
    print(msg)
    print('=' * 100)

    print('First 5 rows of the dataFrame ==>\n')

    print(df.head())

    print('=' * 100)

    print('Last 5 rows of the dataFrame ==>\n')

    print(df.head())

    print('=' * 100)

    print('Info of the dataFrame ==>\n')

    print(df.info())

    print('=' * 100)

    print('Description of the dataFrame ==>\n')

    print(df.describe().T)

    print('#' * 100)
