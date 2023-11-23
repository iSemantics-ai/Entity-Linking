"Utils method for to help utilizing package functionalities "
import os
from typing import Union, Text
from glob import glob
import tarfile
import pandas as pd
import sys
from datetime import datetime
import logging


def current_time():
    return int(datetime.now().strftime('%Y%m%d'))

def read_files(
    path:str, 
    files_pattern:str,
    col_names:list,
    sep:str='\t'
):
    """read and concat list of files

    @params
    -------
    path(str): directory of readable files or path of one file
    files_pattern(str): str expected to be in the files required to read
    col_name(List[str]): columns names in the files
    
    @return
    -------
    pd.DataFrame for all files together
    """
    if os.path.isdir(path):
        files =  glob(f'{path}/*')
    elif os.path.isfile(path):
        return pd.read_csv(path, sep=sep, names=col_names)
    else: 
        raise FileExistsError(f"This path `{path}` not exist")
 
    
    filtered_files = []
    for each_file in files:
        if each_file.__contains__(files_pattern):  #since its all type str you can simply use startswith
            filtered_files.append(each_file)

    print('>> Num of files:',len(filtered_files))
    all_filtered = pd.concat([pd.read_csv(f'{f}',low_memory=False,sep=sep, names=col_names) for f in filtered_files],axis=0)
    all_filtered.drop(all_filtered[all_filtered[col_names[0]] == col_names[0]].index,inplace=True)
    all_filtered= all_filtered.dropna().reset_index(drop=True)    
    return all_filtered


def filter_max(df, indecies ,column='score'):
    """
    Filters a DataFrame by selecting the row with the maximum value in a specified column.

    @params
    -------
    df (pd.DataFrame): The DataFrame to filter.
    indecies(List): list of indecies to filter.
    column (str, optional): The name of the column to use for filtering. Defaults to 'score'.
    
    @return
    -------
    pd.DataFrame: The filtered DataFrame.
    """
    filtered_data = []
    for i, row in df.iterrows():
        # Get the scores for each match
        scores = row[[f"{column}_{x}" for x in indecies]]
        # Get the index of the maximum score
        max_index = scores.to_numpy().argmax()
        # Get the values for the row with the maximum score
        filtered_values = row[row.index[row.index.str.endswith(str(max_index))]]
        # Rename the columns to remove the score index
        filtered_data.append(filtered_values.rename({k: k.replace(f"_{max_index}", '') \
                            for k in filtered_values.index}).to_dict())
    # Convert the filtered data to a DataFrame
    return pd.DataFrame(filtered_data)

def get_console_handler() -> logging.StreamHandler:
    """Get console handler.
    Returns:
        logging.StreamHandler which logs into stdout
    """

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    console_handler.setFormatter(formatter)

    return console_handler

def get_logger(name: Text = __name__, log_level: Union[Text, int] = logging.DEBUG) -> logging.Logger:
    """Get logger.
    
    @params
    -------
    name {Text}: logger name
    log_level {Text or int}: logging level; can be string name or integer value
    
    @return
    -------
    logging.Logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate outputs in Jypyter Notebook
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_console_handler())
    logger.propagate = False

    return logger

def run_athena_query(query_string: str,
              database:str,
              athena_client,
              s3_resource,
              output_bucket: str,
              verbose: bool = True,
              r_type="df"):
    """Run athena query"""

    # submit the Athena query
    if verbose:
        print("Running query:\n " + query_string)

    query_execution = athena_client.start_query_execution(
        QueryString=query_string,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation":f"s3://{output_bucket}/queries/"},
    )
    # wait for the Athena query to complete
    query_execution_id = query_execution["QueryExecutionId"]
    query_state = athena_client.get_query_execution(
        QueryExecutionId=query_execution_id
    )["QueryExecution"]["Status"]["State"]
    while query_state != "SUCCEEDED" and query_state != "FAILED":
        query_state = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )["QueryExecution"]["Status"]["State"]

    if query_state == "FAILED":
        failure_reason = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )["QueryExecution"]["Status"]["StateChangeReason"]
        print(failure_reason)
        df = pd.DataFrame()
        return df
    if r_type == "df":
        ## TODO: fix this to allow user-defined prefix
        results_file_prefix = f"queries/{query_execution_id}.csv"

        filename = f"{query_execution_id}.csv"
        try:
            if verbose:
                print(f"output_bucket: {output_bucket}")
                print(f"results_file_prefix: {results_file_prefix}")
                print(f"filename: {filename}")

            s3_resource.meta.client.download_file(
                Bucket=output_bucket, Key=results_file_prefix, Filename=filename
            )
            df = pd.read_csv(filename)
            if verbose:
                print(f"Query results shape: {df.shape}")
            os.remove(filename)
            s3_resource.meta.client.delete_object(
                Bucket=output_bucket, Key=results_file_prefix
            )
            s3_resource.meta.client.delete_object(
                Bucket=output_bucket, Key=results_file_prefix + ".metadata"
            )
            return df
        except Exception as inst:
            if verbose:
                print(f"Failed download")
                print(f"Exception: {inst}")
            df = None
            pass
    # in case want to return dict not dataf
    if r_type == "dict":
        return athena_client.get_query_results(QueryExecutionId=query_execution_id)

    
def get_top_matches(scores_tuple):
    cand_list = []
    top_score = scores_tuple[0][1]
    for i in range(1, len(scores_tuple)):
        if top_score - scores_tuple[i][1] < 0.01:
            cand_list.append(scores_tuple[i])
        else:
            break
    return cand_list
    

def extract_tar_archive(archive_file, output_folder):
    with tarfile.open(archive_file, "r:gz") as tar:
        tar.extractall(path=output_folder)