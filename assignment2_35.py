import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
import pandas as pd
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    # join product and review data on asin (product id)
    joined = product_data.select('asin').join(review_data, on='asin', how='left')
    
    # groupby asin (product id)
    grouped = joined.groupby(joined['asin']).agg({'reviewerID': 'count', 'overall': 'avg'})
    
    # with and without null values for later calculations
    with_null = grouped.fillna('').fillna(0)
    without_null = grouped.dropna()
    
    # calculations with nulls
    a1 = with_null.agg(F.count(with_null['count(reviewerID)'])).toPandas()
    
    # calculations without null
    a2 = without_null.agg(
        F.avg(without_null['avg(overall)']),
        F.variance(without_null['avg(overall)']),
        F.count(without_null['avg(overall)']),
        F.avg(without_null['count(reviewerID)']),
        F.variance(without_null['count(reviewerID)']),
        F.count(without_null['count(reviewerID)'])
    ).toPandas()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:

    res['count_total'] = int(a1['count(count(reviewerID))'].iloc[0])
    res['mean_meanRating'] = float(a2['avg(avg(overall))'].iloc[0])
    res['variance_meanRating'] = float(a2['var_samp(avg(overall))'].iloc[0])
    res['numNulls_meanRating'] = int(a1['count(count(reviewerID))'].iloc[0]) - int(a2['count(avg(overall))'].iloc[0])
    res['mean_countRating'] = float(a2['avg(count(reviewerID))'].iloc[0])
    res['variance_countRating'] = float(a2['var_samp(count(reviewerID))'].iloc[0])
    res['numNulls_countRating'] = int(a1['count(count(reviewerID))'].iloc[0]) - int(a2['count(count(reviewerID))'].iloc[0])

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_data_flattened = product_data.select(
        product_data[asin_column],
        product_data.categories[0][0].alias(category_column),
        F.explode_outer(product_data.salesRank)
    ).withColumnRenamed('key', bestSalesCategory_column).withColumnRenamed('value', bestSalesRank_column)

    product_data_flattened = product_data_flattened.select(
        [F.when(F.col(c)=='', None).otherwise(F.col(c)).alias(c) for c in product_data_flattened.columns]
    )

    a1 = product_data_flattened.agg(
        F.count(product_data_flattened[asin_column]),
        F.avg(product_data_flattened[bestSalesRank_column]),
        F.variance(product_data_flattened[bestSalesRank_column]),
        F.count(product_data_flattened[category_column]),
        F.countDistinct(product_data_flattened[category_column]),
        F.count(product_data_flattened[bestSalesCategory_column]),
        F.countDistinct(product_data_flattened[bestSalesCategory_column])
    ).toPandas()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:

    res['count_total'] = int(a1[f'count({asin_column})'].iloc[0])
    res['mean_bestSalesRank'] = int(a1[f'avg({bestSalesRank_column})'].iloc[0])
    res['variance_bestSalesRank'] = float(a1[f'var_samp({bestSalesRank_column})'].iloc[0])
    res['numNulls_category'] = res['count_total'] - int(a1[f'count({category_column})'].iloc[0])
    res['countDistinct_category'] = int(a1[f'count(DISTINCT {category_column})'].iloc[0])
    res['numNulls_bestSalesCategory'] = res['count_total'] - int(a1[f'count({bestSalesCategory_column})'].iloc[0])
    res['countDistinct_bestSalesCategory'] = int(a1[f'count(DISTINCT {bestSalesCategory_column})'].iloc[0])

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    # create alias for later use in self-join
    product_data_a = product_data.alias('product_data_a')
    
    # expand also_viwed array into multiple rows
    product_data_a = product_data_a.select(
        product_data_a[asin_column],
        F.explode_outer(product_data_a[related_column][attribute])
    ).withColumnRenamed('col', attribute)
    
    # create alias for later use in self-join
    product_data_b = product_data.alias('product_data_b')
    
    # select just asin and price
    product_data_b = product_data_b.select(
        product_data_b[asin_column],
        product_data_b[price_column]
    ).withColumnRenamed(asin_column, attribute)
    
    # join expanded table with price table
    product_data_flattened = product_data_a.join(product_data_b, on=attribute, how='left')
    
    aggregated = product_data_flattened.groupby(product_data_flattened[asin_column]).agg(
        F.count(product_data_flattened[attribute]),
        F.avg(product_data_flattened[price_column])
    )
    
    # turn count == 0 into null
    aggregated = aggregated.select(
        aggregated[asin_column],
        aggregated[f'avg({price_column})'],
        F.when(F.col(f'count({attribute})')==0, None).otherwise(F.col(f'count({attribute})')).alias(f'count({attribute})')
    )
    
    # final aggregations
    a1 = aggregated.agg(
        F.count(aggregated[asin_column]),
        F.mean(aggregated[f'avg({price_column})']),
        F.variance(aggregated[f'avg({price_column})']),
        F.count(aggregated[f'avg({price_column})']),
        F.mean(aggregated[f'count({attribute})']),
        F.variance(aggregated[f'count({attribute})']),
        F.count(aggregated[f'count({attribute})'])
    ).toPandas()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:

    res['count_total'] = int(a1[f'count({asin_column})'].iloc[0])
    res['mean_meanPriceAlsoViewed'] = float(a1[f'avg(avg({price_column}))'].iloc[0])
    res['variance_meanPriceAlsoViewed'] = float(a1[f'var_samp(avg({price_column}))'].iloc[0])
    res['numNulls_meanPriceAlsoViewed'] = res['count_total'] - int(a1[f'count(avg({price_column}))'].iloc[0])
    res['mean_countAlsoViewed'] = float(a1[f'avg(count({attribute}))'].iloc[0])
    res['variance_countAlsoViewed'] = float(a1[f'var_samp(count({attribute}))'].iloc[0])
    res['numNulls_countAlsoViewed'] = res['count_total'] - int(a1[f'count(count({attribute}))'].iloc[0])

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    aggregated = product_data.select(product_data[price_column]).dropna().summary('mean', '50%').toPandas()
    
    imputed = product_data.select(
        product_data['asin'],
        product_data[price_column].alias(meanImputedPrice_column),
        product_data[price_column].alias(medianImputedPrice_column),
        product_data[title_column].alias(unknownImputedTitle_column)
    )
    
    imputed = imputed.fillna({
        meanImputedPrice_column: aggregated[aggregated['summary'] == 'mean']['price'].iloc[0],
        medianImputedPrice_column: aggregated[aggregated['summary'] == '50%']['price'].iloc[0],
        unknownImputedTitle_column: 'unknown'
    })
    
    a1 = imputed.agg(
        F.count(imputed['asin']),
        F.avg(imputed[meanImputedPrice_column]),
        F.variance(imputed[meanImputedPrice_column]),
        F.count(imputed[meanImputedPrice_column]),
        F.avg(imputed[medianImputedPrice_column]),
        F.variance(imputed[medianImputedPrice_column]),
        F.count(imputed[medianImputedPrice_column]),
    ).toPandas()
    
    a2 = imputed.select(unknownImputedTitle_column).where(imputed[unknownImputedTitle_column] == 'unknown').count()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:

    res['count_total'] = int(a1['count(asin)'].iloc[0])
    res['mean_meanImputedPrice'] = float(a1[f'avg({meanImputedPrice_column})'].iloc[0])
    res['variance_meanImputedPrice'] = float(a1[f'var_samp({meanImputedPrice_column})'].iloc[0])
    res['numNulls_meanImputedPrice'] = res['count_total'] - int(a1[f'count({meanImputedPrice_column})'].iloc[0])
    res['mean_medianImputedPrice'] = float(a1[f'avg({medianImputedPrice_column})'].iloc[0])
    res['variance_medianImputedPrice'] = float(a1[f'var_samp({medianImputedPrice_column})'].iloc[0])
    res['numNulls_medianImputedPrice'] = res['count_total'] - int(a1[f'count({medianImputedPrice_column})'].iloc[0])
    res['numUnknowns_unknownImputedTitle'] = int(a2)

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_processed_data_output = product_processed_data.select(
        product_processed_data['asin'],
        F.split(F.lower(product_processed_data[title_column]), ' ').alias(titleArray_column),
    )
    
    word2Vec = M.feature.Word2Vec(vectorSize = 16, minCount = 100, numPartitions = 4, seed = SEED, inputCol = titleArray_column, outputCol = titleVector_column)
    
    model = word2Vec.fit(product_processed_data_output)


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:

    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------

    stringIndexer = M.feature.StringIndexer(inputCol = category_column, outputCol = categoryIndex_column)
    
    model = stringIndexer.fit(product_processed_data)
    
    product_processed_data_output = model.transform(product_processed_data)
    
    ohe = M.feature.OneHotEncoderEstimator(inputCols = [categoryIndex_column], outputCols = [categoryOneHot_column], dropLast = False)
    
    model = ohe.fit(product_processed_data_output)
    
    product_processed_data_output = model.transform(product_processed_data_output)
    
    pca = M.feature.PCA(k=15, inputCol = categoryOneHot_column, outputCol = categoryPCA_column)
    
    model = pca.fit(product_processed_data_output)
    
    product_processed_data_output = model.transform(product_processed_data_output)
    
    a = product_processed_data_output.agg(
        F.count(product_processed_data_output['asin']),
        M.stat.Summarizer.mean(product_processed_data_output[categoryOneHot_column]),
        M.stat.Summarizer.mean(product_processed_data_output[categoryPCA_column])
    ).toPandas()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:

    res['count_total'] = int(a['count(asin)'].iloc[0])
    res['meanVector_categoryOneHot'] = [float(x) for x in a[f'mean({categoryOneHot_column})'].iloc[0]]
    res['meanVector_categoryPCA'] = [float(x) for x in a[f'mean({categoryPCA_column})'].iloc[0]]

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    dt = M.regression.DecisionTreeRegressor(labelCol = 'overall', maxDepth = 5)
    
    model = dt.fit(train_data)
    
    result = model.transform(test_data)
    
    evaluator = M.evaluation.RegressionEvaluator(labelCol = 'overall')
    
    rmse = evaluator.evaluate(result)
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    
    res['test_rmse'] = float(rmse)

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    evaluator = M.evaluation.RegressionEvaluator(labelCol = 'overall')
    
    rmses = []
    
    depths = [5, 7, 9, 12]
    
    for tree_depth in depths:
        new_train, validation = train_data.randomSplit([.75, .25])
        
        dt = M.regression.DecisionTreeRegressor(labelCol = 'overall', maxDepth = tree_depth)
    
        model = dt.fit(new_train)
        
        result = model.transform(validation)
        
        rmse = evaluator.evaluate(result)
        
        rmses.append(rmse)
    
    best_depth = depths[rmses.index(min(rmses))]
    
    dt = M.regression.DecisionTreeRegressor(labelCol = 'overall', maxDepth = best_depth)
    
    model = dt.fit(train_data)
    
    result = model.transform(test_data)
    
    rmse = evaluator.evaluate(result)
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:

    res['test_rmse'] = float(rmse)
    res['valid_rmse_depth_5'] = float(rmses[0])
    res['valid_rmse_depth_7'] = float(rmses[1])
    res['valid_rmse_depth_9'] = float(rmses[2])
    res['valid_rmse_depth_12'] = float(rmses[3])
    
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

