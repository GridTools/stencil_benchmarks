import pandas as pd


def expand_tuples(data_frame):
    if all(dtype.kind != 'O' for dtype in data_frame.dtypes.values):
        return data_frame
    out = pd.DataFrame()
    for column, dtype in data_frame.dtypes.iteritems():
        if dtype.kind == 'O':
            expanded = pd.DataFrame(data_frame[column].tolist(),
                                    index=data_frame.index)
            out[[f'{column}-{i}' for i in expanded.columns]] = expanded
        else:
            out[column] = data_frame[column]
    return out


def write_csv(data_frame, filename):
    expand_tuples(data_frame).to_csv(filename)


def read_csv(filename):
    return pd.read_csv(filename, index_col=0)
