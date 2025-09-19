import pandas as pd

root_path = f'../data/nhanes/data/'
def concat_data_across_years(data_type, file_code, years, year_char):
    """
    This function concat data across years.

    :param data_type: The NHANES data type (demographic, dietary, examination, laboratory, questionnaire)
    :param file_code: the NHANES code of the file.
    :param years: A list of strings identifies which year the data is from. E.g. ['0102', '0304', ..., '1314']
    :param year_char: The char NHANES data used to identify years. This char attaches to the file names as suffix.
        For 01-02 data, the char is B; For 03-04 data, the char is C.
    :return: Returns a pandas dataframe of the concatenated data
    """
    df = pd.DataFrame()
    for year in years:
        if year == '1720':
            path = root_path + year + '/' + data_type + '/P_' + file_code + '.xpt'
        else:
            path = root_path + year + '/' + data_type + '/' + file_code + '_' + year_char + '.xpt'

        df_temp = pd.read_sas(path, encoding='ISO-8859-1')
        year_char = chr(ord(year_char) + 1)
        # Record which year the data comes from
        df_temp['years'] = year

        if df.empty:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp])

    # pandas read_sas has an issue to read 0 as 5.397e-79 ...
    df.replace(5.397605346934028e-79, 0, inplace=True)
    return df

def merge_with_or(df1, df2):
    """
    Merges two DataFrames on a specified key(s) and combines shared columns using an 'OR' relation.

    Parameters:
    - df1, df2: DataFrames to be merged.

    Returns:
    - A merged DataFrame with combined shared columns using an 'OR' logic.
    """
    # Merge the DataFrames
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='left', suffixes=('_df1', '_df2'))

    # Find shared columns, excluding the key(s) used for merging
    shared_columns = set(df1.columns) & set(df2.columns)

    for col in shared_columns:
        col_df1 = f'{col}_df1'
        col_df2 = f'{col}_df2'

        # Apply 'OR' operation for the shared column and assign it to the merged DataFrame
        # Only df2 contains NaN values, so we need to fill them with False before converting to int
        merged_df[col] = (merged_df[col_df1] | merged_df[col_df2].fillna(False).astype(int))

        # Drop the original columns from the merge
        merged_df.drop(columns=[col_df1, col_df2], inplace=True)

    return merged_df