import pandas as pd


cps = [f'core{i}' for i in range(1,11) ]
column_names = ['date', 'all'] + cps
print(column_names)
df = pd.read_csv('/home/vito/uspex_python/temperature_log.txt', header=None, names=column_names, index_col=False)

def celcius_to_float(temp_str:str):
    temp = float(temp_str[1:5])
    return(temp)

other_columns = column_names[1:]

# Apply a function (e.g., squaring values) to the selected columns
# df[other_columns] = df[other_columns].apply(lambda x: celcius_to_float(x))

print(df[other_columns])
