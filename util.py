import numpy as np
import pandas as pd
import re
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def clean(file):
    # change to lower case
    df = pd.read_csv(file, dtype = str)
    for col in df.columns.values:
        df[col] = df[col].str.lower()

    # remove special characters in the offer column
    if file == 'offer_retailer.csv':
        for index, row in df.iterrows():
            txt = row['OFFER']
            df.loc[index, "OFFER"] = re.sub(r"[^a-z0-9 ]", "", txt)
    # remove duplicate rows
        df.drop_duplicates(subset=['OFFER'], inplace = True)
    return df


def loadData():
    offer = clean('offer_retailer.csv')
    offer.replace(np.nan,'',regex=True, inplace=True)

    brand = clean('brand_category.csv')
    brand.rename(columns={"BRAND_BELONGS_TO_CATEGORY": "CATEGORY"}, inplace=True)
    brand['RECEIPTS'] = pd.to_numeric(brand['RECEIPTS'])

    category = clean('categories.csv')
    category.rename(columns={"PRODUCT_CATEGORY": "CATEGORY",
                          "IS_CHILD_CATEGORY_TO": "PARENT_CATEGORY"}, inplace=True)
    category.drop(columns=['CATEGORY_ID'], inplace = True)

    # join the brand and category table, using pandas merge function which is equal to SQL inner join
    brand_cat = brand.merge(category, right_on = ['CATEGORY'], left_on = ['CATEGORY'])
    full = offer.merge(brand_cat, how = 'left', right_on = ['BRAND'], left_on = ['BRAND'])
    full.replace(np.nan,'',regex=True, inplace=True)
    # rearrange columns
    cols = ['OFFER', 'RETAILER', 'BRAND', 'CATEGORY', 'PARENT_CATEGORY', 'RECEIPTS']
    full = full[cols]
    return offer, brand, category, full


def a():
    print('a')

def categorySearch(queryLst: list, offer, brand, category, full, N = 20):
    """
    query: a list of string inputs of search in category. length of the list is n.
    N: top N offers selected by the matching. default 20.
    return: n csv files stored in the folder, each stores the top N selected matching for the corresponding query.
    """
    model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    catDict = full['CATEGORY'].to_list()
    catDict_vecs = model.encode(catDict)
    similarity = full.copy()
    query = model.encode(queryLst)
    cos = cosine_similarity(query, catDict_vecs).round(4)
    for i, q in enumerate(queryLst):
        similarity[q] = cos[i].reshape(-1, 1)
    df = similarity.copy()

    # Display the result, top N matched offer
    for q in queryLst:
        df.drop_duplicates(subset=['OFFER'], inplace = True)
        a = df.sort_values(by=[q, 'RECEIPTS'], ascending=[False, False])
        a.rename(columns={q: "SCORE"}, inplace=True)
        # top N offers
        a = a.iloc[:N][['OFFER', 'RETAILER', 'BRAND', 'CATEGORY', 'RECEIPTS', 'SCORE']]
        fname = f'Category Search Top {N} {q}.csv'
        a.to_csv(fname)
        print(f'The output csv file {fname} is in the folder')

def brandSearch(query: str, offer, brand, category, full, cutoff = 0.9, N = 10, ):
    """
    Searching the brand in the joined full table.
    query: a single string input of search in brand
    return: a dataframe with the last column is the similarity score between the query and brand in each row
    """
    model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    query = query.lower()
    brandDict = full['BRAND'].to_list()
    brandDict_vecs = model.encode(brandDict)
    similarity = full.copy()
    q = model.encode([query])
    cos = cosine_similarity(q, brandDict_vecs).round(4)
    similarity['SCORE'] = cos.reshape(-1, 1)
    df = similarity.copy()
    df.sort_values(by=['SCORE', 'RECEIPTS'], ascending=[False, False])
    brandOut = df[df['SCORE'] >= cutoff]

    # case one: if the brand search can identify something
    if brandOut.shape[0] > 0:
        brandOut.drop_duplicates(subset=['OFFER'], inplace = True)
        # top N offers
        b = brandOut.iloc[:N][['OFFER', 'RETAILER', 'BRAND', 'CATEGORY', 'RECEIPTS', 'SCORE']]
        fname = f'Brand Search Top {N} {query}.csv'
        # print the result here, and save to a csv file
#         print(fname)
#         print(b)
        b.to_csv(fname)
    # case two: if the brand search cannot identify any matching
    else:
        print('No good matching found in the offer table. Searching the brand_category list instead. ')
        brandOut = brandToCatSearch(query, offer, brand, category, full)
        if brandOut.shape[0] == 0:
            print('No matching found, please double check your input')
            return
        return brandOut

def brandToCatSearch(query, offer, brand, category, full, N = 30, k = 3):
    """
    Searching the brand in the brand_category table instead.
    query: a string input of search in brand
    return: a dataframe with the last column is the similarity score between the query and brand in each row
    """
    model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    brandDict = brand['BRAND'].to_list()
    brandDict_vecs = model.encode(brandDict)
    similarity = brand.copy()
    q = model.encode([query])
    cos = cosine_similarity(q, brandDict_vecs).round(4)
    similarity[query] = cos.reshape(-1, 1)
    df = similarity.copy()
    sortdf = df.sort_values(by=[query, 'RECEIPTS'], ascending=[False, False])

    match = sortdf.iloc[:k]
    # this match returns the top k matched category, by searching in the brand-category table
    print(f'Top {k} matched category')
    print(match)

    queryLst, scores = match['CATEGORY'].to_list(), match[query].to_list()

    catDict = full['CATEGORY'].to_list()
    catDict_vecs = model.encode(catDict)
    similarity = full.copy()
    query = model.encode(queryLst)
    cos = cosine_similarity(query, catDict_vecs).round(4)
    for i, q in enumerate(queryLst):
        similarity[q] = cos[i].reshape(-1, 1)
    df = similarity.copy()

    df[queryLst] = df[queryLst]*scores
    df[queryLst].round(4)
    df['SCORE'] = df[queryLst].max(axis=1)
    df.drop(columns=queryLst, inplace = True)
    df.drop_duplicates(subset=['OFFER'], inplace = True)

    sortdf = df.sort_values(by=['SCORE', 'RECEIPTS'], ascending=[False, False])

    out = sortdf.iloc[:N]
    fname = f'Brand Search through Category - Top {N}_{q} Match.csv'
    out.to_csv(fname)
    print()
    print(fname)
    print(out)
    return out

def retailerSearch(query, offer, brand, category, full, N = 10):
    """
    query: a single string inputs of search in retailer.
    N: top N offers selected by the matching. default 10.
    return: a csv files stores the top N selected retailer matching.
    """
    model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    retDict = full['RETAILER'].to_list()
    retDict_vecs = model.encode(retDict)
    similarity = full.copy()
    q = model.encode([query])
    cos = cosine_similarity(q, retDict_vecs).round(4)
    similarity['SCORE'] = cos.reshape(-1, 1)
    df = similarity.copy()

    # Display the result, top N matched offer
    df.drop_duplicates(subset=['OFFER'], inplace = True)
    a = df.sort_values(by=['SCORE', 'RECEIPTS'], ascending=[False, False])
    # top N offers
    a = a.iloc[:N][['OFFER', 'RETAILER', 'BRAND', 'CATEGORY', 'RECEIPTS', 'SCORE']]
    fname = f'Retailer Search Top {N} {query}.csv'
    print()
    print(fname)
    print(a)
    
    a.to_csv(fname)
