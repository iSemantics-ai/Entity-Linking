import re
import string
import pandas as pd
from math import ceil
from tqdm.auto import tqdm
import boto3
from typing import List, Dict
from itertools import compress
from collections import defaultdict
def extract_prefix(text,
                   add:str="bi#",
                   prefix_len:int=2):
    addition = ''
    clean_text = re.sub(f"[{re.escape(string.punctuation)}]",
                        "",
                        text.strip()).lower()
    words = text.lower().split(" ")
    first_word = words[0].strip()
    if first_word == "the" and len(words) > 1:
        first_word = words[1]
        addition = 'the '

    text_prefix = add + clean_text[len(addition):].replace(" ", "")[:prefix_len]
    return text_prefix, addition

def create_prefix_query(names,
                        prefix_len,
                        pk_mark:str='bi#',
                        sk_mark:str='n#',
                        all_words:bool=False):
    """
    Create a prefix query for a list of company names.

    :param names: A list of company names.
    :param add: An optional prefix to add to each company name.
    :return: A dictionary of prefix queries, where each key is a prefix and each value is a set of company names.
    """
    queries = []
    for i, company in enumerate(names):
        company_prefix, addition = extract_prefix(company,
                                                  add=pk_mark,
                                                  prefix_len=prefix_len)
        words = company.lower().split(" ")
        if all_words:
            queries.append((company_prefix,f"n#{company.lower()}", i))
        else:
            sort_values = ["n#{}".format(" ".join([words[0], words[1][:2]]))\
                           if i  == len(words) -2 else \
                           "n#{}".format(" ".join(words[:len(words)-i]))
                                 for i in range(len(words))]
            if addition != '':
                sort_values = sort_values[:-1]
                wo_addition = company.lower().replace("the", '').strip().split(" ")
                
                sort_values += ["n#{}".format(" ".join([wo_addition[0], wo_addition[1][:2]]))\
                                if i  == len(wo_addition) -2 else \
                                "n#{}".format(" ".join(wo_addition[:len(wo_addition)-i]))
                                for i in range(len(wo_addition))]
                
                sort_values[len(words)-1] = sort_values[len(words)-1].strip()
            
            sort_values =[(company_prefix, x, i) for x in sort_values]
            queries += sort_values
    # Create DataFrame from the queries list
    queries = pd.DataFrame(queries, columns=["pk", "sk", "word_idx"])\
                    .drop_duplicates("sk").sort_values("sk", ascending=False)
    return queries

def run_dynamo_query(dynamo_client,
                     table_name,
                     index_name,
                     key_condition_expression,
                     exp_attribute_values,
                     required_values,
                    max_pages):
    t_results = []
    pages = 0
    response = dynamo_client.query(
        TableName=table_name,
        IndexName=index_name,
        KeyConditionExpression=key_condition_expression,
        ExpressionAttributeValues=exp_attribute_values,
    )
    t_results.extend([tuple([x[k][v] for k,v in required_values.items()]) for x in response['Items']])
    while 'LastEvaluatedKey' in response and pages > max_pages:
        response = dynamo_client.query(
        TableName=table_name,
        IndexName=index_name,
        KeyConditionExpression=key_condition_expression,
        ExpressionAttributeValues=exp_attribute_values,
        ExclusiveStartKey=response['LastEvaluatedKey'])
        t_results.extend([tuple([x[k][v] for k,v in required_values.items()]) for x in response['Items']])
        pages += 1
    return set(t_results)

def query_dynamodb_table(
    dynamodb,
    table_name: str,
    index_name: str,
    attribute_name: str,
    sort_name: str,
    query_values: pd.DataFrame,
    sort: bool = True,
    condition='begins_with',
    max_pages=30
) -> List[Dict]:

    """
    Queries a DynamoDB table for a list of query values in the specified attribute.

    Args:
        dynamodb (boto3.client): A DynamoDB client instance.
        table_name (str): The name of the DynamoDB table to query.
        index_name (str): The name of the secondary index to use for the query.
        attribute_name (str): The name of the attribute to search for the query values.
        sort_name (str): The name of the attribute to sort the results by.
        query_values (List[str]): A list of values to query for in the specified attribute.
        sort_limit (int, optional): The maximum number of sort values to filter for. Defaults to 20.
        sort (bool, optional): Whether or not to perform a sort. Defaults to False.

    Returns:
        List[Dict]: A list of items that match the query values.
    """
    # Initialize an empty list to store the results
    t_results = set()
    processed = []
    # Define two different key condition expressions depending on whether or not a sort is performed
    if condition == "begins_with":
        key_condition_expression_1 = f"{attribute_name} = :val1 and begins_with({sort_name}, :val2)"
    elif condition == "equal":
        key_condition_expression_1 = f"{attribute_name} = :val1 and {sort_name} = :val2"
    # Loop through each query value and its associated sort values
    with tqdm(query_values.groupby('pk'), unit="index", colour='MAGENTA') as query_key:
        for i, pk_group in query_key:
            query_key.set_description(f"{i}: {len(pk_group)}")
            for x, row in pk_group.iterrows():
                if not row['word_idx'] in processed:
                    exp_attribute_values = {":val1": {"S": i}, ":val2": {"S": row['sk']}}
                    response = run_dynamo_query(dynamodb,
                                    table_name,
                                    index_name,
                                    key_condition_expression_1,
                                    exp_attribute_values,
                                    {"normalizedname":"S", "companyid":"S"},
                                    max_pages
                                    )
                    if len(response) > 0:
                        processed.append(row['word_idx'])
                    t_results = t_results | response
    return pd.DataFrame(t_results, columns=["text", "ids"]) 
# Function to scan the entire table and collect data
def scan_table(dynamo_client: boto3.client, table_name: str):
    data = []
    response = dynamo_client.scan(TableName=table_name)
    data.extend(response['Items'])
    
    while 'LastEvaluatedKey' in response:
        response = dynamo_client.scan(TableName=table_name, ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    
    return data

def batch_write_dynamodb_items(
    dynamo_client: boto3.client,
    dynamo_items: List[dict],
    table_name: str,
) -> None:
    """
    Splits a list of DynamoDB items into batches of 25 and inserts them into a DynamoDB table using the batch_write_item() method.

    Args:
        dynamo_items (List[dict]): A list of DynamoDB items to be inserted into the table.
        table_name (str): The name of the DynamoDB table.
        dynamo_client (boto3.client): A Boto3 client object for communicating with DynamoDB.

    Returns:
        None: This function does not return anything, it simply inserts items into the table.

    Raises:
        None: This function does not raise any exceptions.

    """
    # Split the items into batches of 25, the maximum batch size for DynamoDB
    batches = [dynamo_items[i : i + 25] for i in range(0, len(dynamo_items), 25)]

    # Insert each batch using the batch_write_item() method
    failed_batches = []
    for batch in tqdm(
        batches,
        total=len(batches),
        desc=f"Batch insertion into `{table_name}` Table",
    ):
        request_items = {
            table_name: [{"PutRequest": {"Item": item}} for item in batch]
        }
        try:
            response = dynamo_client.batch_write_item(RequestItems=request_items)
        except Exception as e:
            request_items["Exception"] = e
            failed_batches.append(request_items)
    print(f"Success rate {len(batches) - len(failed_batches)}/{len(batches)} ")
    
    return failed_batches


def batch_write(table, items):
    """
    Write a batch of items to a DynamoDB table.

    @params
    -------
    table (boto3.resource): A resource representing an Amazon DynamoDB table.
    items (list): A list of items to write to the table.

    @returns
    --------
    failed_batches(list): This list contains all the exceptions of put_item requests.

    @raises
    -------
    Exception: If any errors occur while writing the batch.
    """
    # Insert each batch using the batch_write_item() method
    failed_batches = []
    with table.batch_writer() as batch:
        for item in tqdm(items, total=len(items), desc="Put Items"):
            try:
                response =  batch.put_item(Item=item)
            except Exception as e:
                failed_batches.append({"item":item, "exception": e})

    print(f"Success rate {len(items) - len(failed_batches)}/{len(items)} ")
    return failed_batches