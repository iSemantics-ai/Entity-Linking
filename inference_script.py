import faiss
import boto3
import os
import pandas as pd
import torch
from io import StringIO
from pathlib import Path
import json
import re
from tqdm.auto import tqdm
import operator
from matcher.utils import (read_files,
                           get_logger,
                           filter_max,
                           current_time,
                           extract_tar_archive,
                          run_athena_query)
from matcher.dynamo_utils import (create_prefix_query,
                                  batch_write,
                                  scan_table,
                                 query_dynamodb_table)
from matcher.core import SimCSE_Matcher
from distances import FuzzyWuzzyTokenSort
content = open('Scala_Patterns/bussiness_types.txt', 'r').read()
# Define all the business types
b_types = re.findall(r'=\s*"([^"]+)"', content)
b_types = set(map(lambda x : x.lower(), b_types))
print("StandardGpuResources", hasattr(faiss, "StandardGpuResources"))
print("cuda", torch.cuda.is_available())


########################## ENV Variables ##########################
# Initialize the dynamo client
dynamodb = boto3.client(
    "dynamodb", region_name=os.getenv("AWS_REGION") or "us-east-1")
dynamo_resource = boto3.resource('dynamodb', region_name=os.getenv("AWS_REGION") or "us-east-1")
# Initialize the S3 client
s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION") or "us-east-1")

s3_resource = boto3.resource(
    "s3",
    region_name=os.getenv("AWS_REGION") or "us-east-1")
# Athena client
athena_client = boto3.client(
            "athena",
            region_name=os.getenv("AWS_REGION") or "us-east-1"
        )
# initiate logger object
logger = get_logger(f'Entity_linking', 'INFO')
# redis_client = pyredis.Redis(host=os.getenv("REDIS_HOST") or "localhost",
#                              port=os.getenv("REDIS_PORT") or 6379,
#                              db=os.getenv("REDIS_DB") or 5,
#                              decode_responses=True)

models_dir = Path("models_data")
Path.mkdir(models_dir, exist_ok="True")

links_meta = dict(
    table_name=os.getenv("LINKS_TABLE") or "test-prod-entity-links",
    query_feature="inferess_name",
    query_id="inferess_entity_id",
    source_feature="sp_name",
    source_id="sp_id",
    score="match_score")

##################################################################
model_meta = dict(bucket_name="ecomap-dl-pipeline",
                  models_keys={
                      "general_encoder": "name_matching/artifacts/all-MiniLM-Pos.tar.gz"
                      })

query_meta = dict(data_type="s3",
                  bucket_name=os.getenv("QUERY_BUCKET") or "inferess-data",
                  file_key=os.getenv("QUERY_KEY") or \
                           f"ecomap/ecomap-daily-updates-partitioned/company/filedasofdate",
                  feature=os.getenv("QUERY_FEATURE") or "normalized_name",
                  ids=os.getenv("QUERY_ID") or "inferess_entity_id",
                  cleaning={os.getenv("CLEAN") or "undisclosed": ['true', True]},
                  candidate_path="name_matching/candidate_matches",
                  drop_prematch=os.getenv("DROP_PRE") or True,
                  
                  )

source_meta = dict(lookup_table=os.getenv("SOURCE_DB") or "company",
                   index_column=os.getenv("INDEX_COLUMN") or "gsi",
                   attribute_name=os.getenv("ATTR_NAME") or "gsipk",
                   _add=os.getenv("PREFIX_ST") or "bi#",
                   prefix_len=2,
                   sort_key=os.getenv("SORT_KEY") or "gsisk",
                   normalized_column=os.getenv("NORM_COL") or "normalizedname",
                   id_column=os.getenv("ID_COL") or "companyid",
                   database_type="dynamodb",  # dynamodb|local|s3
                   local_info={
                       'source_dir': 'data/sp-crdb/normalized-company.csv',
                       'files_pattern': 'normalized',
                       'columns_names': ['companyid', 'companyname'],
                       'source_id': 'companyid',
                       'source_feature': 'companyname'
                   },
                   sort=True)


thresholds = dict(match_thresh=os.getenv("MATCH_THRESH") or 0.973,
                  cand_thresh=os.getenv("CANDIDATE_MATCH") or 0.5,
                  top_k=30)



def main():
    ########################## Load matchers from S3 ##########################
    # Download and use given matchers
    logger.info("loading matchers artifacts")
    matchers = dict()
    for i, model_key in model_meta['models_keys'].items():
        downloaded_file = models_dir / Path(model_key).name
        s3.download_file(model_meta['bucket_name'], model_key, str(downloaded_file))
        print("Downloading `s3://{}/{}".format(model_meta['bucket_name'], model_key))
        extract_tar_archive(downloaded_file, downloaded_file.parent)
        print("Extract model at `{}`".format(downloaded_file.parent))
        matchers[i] = SimCSE_Matcher(str(models_dir / downloaded_file.name.split('.')[0]))

    ########################## Read Query(Inferess) And Source(S&P) Data ##########################
    logger.info("scanning pre-matched names")
    # Read pre-matched data
    logger.info("Reading {} table".format(links_meta['table_name']))
    links_table = pd.DataFrame(scan_table(dynamo_client=dynamodb,
                                         table_name=links_meta['table_name']))
    # read links from athena
    inferess_links= run_athena_query(query_string='''SELECT inferess_entity_id FROM "company_reference_db"."inferess_links"''',
              database="company_reference_db",
              athena_client=athena_client,
              s3_resource=s3_resource,
              output_bucket=query_meta['bucket_name'],
              verbose=False,
              r_type="df"
              )

    if len(links_table) > 0 :
        links_table[links_meta['query_feature']] = links_table[links_meta['query_feature']].apply(lambda x : x["S"])
        links_table[links_meta['source_feature']] = links_table[links_meta['source_feature']].apply(lambda x: x["S"])
        pre_linked = links_table[links_meta['query_feature']].tolist()
    else:
        pre_linked = []


    # If the data type is 'local'
    if query_meta['data_type'] == 'local':
        # Read the query data from a CSV file and drop any duplicate values based on a specified feature
        query_data = pd.read_csv(query_meta['file_key'], sep='\t').drop_duplicates(query_meta['feature'])
    elif query_meta['data_type'] == 's3':
        # # Construct the S3 URL
        # s3_url = f"s3://{query_meta['bucket_name']}/{query_meta['file_key']}"
        # # Read the TSV file from S3 using pandas
        # query_data = pd.read_csv(s3_url, sep='\t').drop_duplicates(query_meta['feature'])
        # Get the object from S3
        logger.info("reading query data from path: s3://{}/{}" \
                    .format(query_meta['bucket_name'],
                            query_meta['file_key']))
        objects = s3.list_objects_v2(Bucket=query_meta['bucket_name'],
                                     Prefix=query_meta['file_key'])
        query_data = pd.DataFrame()
        if objects['KeyCount'] > 0:
            for key in objects['Contents']:
                response = s3.get_object(Bucket=query_meta['bucket_name'], Key=key['Key'])
                content = response['Body'].read().decode('utf-8')
                query_data = pd.concat([query_data,pd.read_csv(StringIO(content), sep='\t')], axis=0)
        logger.info("query_data.columns: {}".format(query_data.columns))
        # Convert the content to a Pandas DataFram
        query_data = query_data.drop_duplicates([query_meta['feature']]).reset_index(drop=True)

        query_data.set_index(query_meta['feature'], inplace=True)
        query_len = len(query_data.index)

    if query_meta['drop_prematch']:
        logger.info("drop pre-linked entities")
        # Drop the pre_linked entit
        query_data = query_data[~query_data.index.isin(pre_linked)]
        # Drop exist links from Inferess_links
        query_data = query_data[~query_data.inferess_entity_id.isin(inferess_links.inferess_entity_id)]
        remain_len = len(query_data.index)
        logger.info(f"Match with {query_len - remain_len} name, remaining names to be matched: {remain_len}")

    if query_meta.get('cleaning'):
        # Iterate over each cleaning item and remove any rows that match the specified values
        for k, v in query_meta['cleaning'].items():
            query_data.drop(query_data[query_data[k].isin(v)].index, inplace=True)

    # Convert the feature column to a list and create prefix queries for each value
    all_companies = query_data.index.tolist()
    queries = create_prefix_query(all_companies,
                                  prefix_len=source_meta['prefix_len'], 
                                  pk_mark=source_meta['_add'],
                                  sk_mark='n#', 
                                  all_words=False)
    ########################## Query Source Data ###############################
    # Load business types
    if source_meta['database_type'] == 'local':
        source_frame = read_files(source_meta['local_info']['source_dir'],
                                  source_meta['local_info']['files_pattern'],
                                  source_meta['local_info']['columns_names'],
                                  sep='\t')
        source_frame = source_frame.drop_duplicates([source_meta['local_info']['source_feature']])
        source_frame = source_frame.dropna().reset_index(drop=True)
        source_frame.rename(columns={source_meta['local_info']['source_feature']: "text",
                                     source_meta['local_info']['source_id']: "ids"}, inplace=True)
    elif source_meta['database_type'] == "dynamodb":
        logger.info("query prefix indecies within {} table" \
                    .format(source_meta['lookup_table']))
        # Query source data with prefix indecies
        source_frame = query_dynamodb_table(
            dynamodb=dynamodb,
            table_name=source_meta['lookup_table'],
            index_name=source_meta['index_column'],
            attribute_name=source_meta['attribute_name'],
            sort_name=source_meta['sort_key'],
            query_values=queries,
            sort=source_meta['sort'],
            condition = "begins_with",
            max_pages=50
        )
        source_frame = source_frame.drop_duplicates(['text']).reset_index(drop=True)
        logger.info("number of selected entities for matching is {}" \
                    .format(len(source_frame)))

        
    ########################## Match with Atention Matcher ###############################
    logger.info("start embedding and matching")
    matches =  matchers['general_encoder'].match_data(source_text=source_frame['text'].tolist(),
                                        source_ids=source_frame['ids'].tolist(),
                                        query_text=query_data.index.tolist(),
                                        b_size=500_000,
                                        top_k=thresholds['top_k'],
                                        threshold=thresholds['cand_thresh'])
    
    match_scope = []
    for i, row in matches.iterrows():
        match_scope.append({"query_text":row['query'], "source_text":row['match_text'], "sim_score":row['score']})
        # TODO Define function to return all the names with score close to top_score
        close_matches = get_top_matches(row['matches']) 
        for match_row in close_matches:
            match_scope.append({"query_text":row['query'] ,
                                "source_text":match_row[0],
                                "sim_score":match_row[1],
                                 })
    match_scope = pd.DataFrame(match_scope)
    match_scope.loc[:, "diff"] =  match_scope.apply(lambda x :\
                                    set(filter(None,
                                               set(x['query_text'].lower().split()) ^\
                                               set(x['source_text'].lower().split()))), axis=1) 
    match_scope.loc[:, 'bt_addition'] = match_scope['diff'].apply(lambda x: True if  len(x - b_types) == 0 else False)
    # Filter Step 1: Pickup all names with score > 0.9 OR bt_addition is True
    match_scope = match_scope.query("sim_score > 0.9 or bt_addition == True").reset_index(drop=True)
    
    ################## Using fuzzy matching to score Char-Based similarity on the match_scope data #########
    # Intitialize the Fuzzy macther
    fuzzy_matcher = FuzzyWuzzyTokenSort()
    match_scope["fuzzy_score"] = match_scope.apply(lambda x: fuzzy_matcher.sim(x['query_text'],
                                             x['source_text']), axis=1)
    
    
    accepted_1 = match_scope.query("sim_score > 0.98 and bt_addition == True and fuzzy_score > 0.9").reset_index(drop=True)
    match_scope.drop(accepted_1.index, inplace=True)
    accepted_2 = match_scope.query("sim_score > 0.99")
    match_scope.drop(accepted_2.index, inplace=True)
    
    accepted = pd.DataFrame(\
                pd.concat([accepted_1, accepted_2], axis=0).groupby('query_text')\
                .apply(lambda x: max(x.to_dict("records"),
                key=operator.itemgetter('fuzzy_score'))).tolist()
                )
	if len(accepted) > 0:	
        # Saving process should be by get the names from source table then save multiple ids if exists
        prefix= create_prefix_query(accepted['source_text'].tolist()
                                     ,source_meta['prefix_len'],
                                     pk_mark=source_meta['_add'],
                                     sk_mark='n#',
                                     all_words=True)
        # Query source data with prefix indecies
        sf = query_dynamodb_table(
            dynamodb=dynamodb,
            table_name=source_meta['lookup_table'],
            index_name=source_meta['index_column'],
            attribute_name=source_meta['attribute_name'],
            sort_name=source_meta['sort_key'],
            query_values= prefix,
            sort=source_meta['sort'],
            condition='equal'
        )

        sf.set_index("text", inplace=True)

        source_ids = []
        for i, row in tqdm(accepted.iterrows(), total=len(accepted)):
            if row['source_text'] in sf.index.tolist():
                match = sf.loc[row['source_text']]
                match_id= match['ids']
                if not isinstance(match_id, str):
                    match_id = set(match_id)
                else:
                    match_id = {match_id}
            else:
                print(row['sp_name'], row['sp_id'] , '\n',row['inferess_name'] ,'\n')
            source_ids.append(match_id)

        accepted["query_id"] = query_data.loc[accepted.query_text].inferess_entity_id.tolist()
        accepted['source_ids'] = source_ids

         ################# Ingest the accepted data ####################
        items = []
        for i, row in accepted.iterrows():
            for sp_id in row['source_ids']:
                items.append({
                    "pk": f"id#{str(row['query_id'])}",
                    "sk": f"{row['source_text']}#{sp_id}",
                    links_meta['query_id']: str(row['query_id']),
                    links_meta['query_feature']: row['query_text'],
                    links_meta['source_id']: str(sp_id),
                    links_meta['source_feature']: row['source_text'],
                    links_meta['score']: str(round(float(row['sim_score']), 3))

                })
        for k, v in items[0].items():
            assert isinstance(k, str)
        i_frame = pd.DataFrame(items)
        assert i_frame.apply(lambda x : x['sp_id'] == x['sk'].split("#")[-1], axis=1).sum() == len(i_frame)
        assert i_frame.apply(lambda x : x['sp_name'] == ''.join(x['sk'].split("#")[:-1]), axis=1).sum() == len(i_frame)
        assert i_frame.apply(lambda x : x['inferess_entity_id'] == x['pk'].split("#")[-1], axis=1).sum() == len(i_frame)
        table = dynamo_resource.Table("test-prod-entity-links")
        failed_batches = batch_write(table, items)
    
    # TODO: Saving match scope
    
    return True


if __name__ == "__main__":
    main()
