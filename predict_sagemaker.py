
# coding: utf-8

# In[69]:


import pandas as pd 
import hashlib
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from utils import logger
#def lassoSelection(X,y,)
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import time

def scores_for_ks(test_labels, knn_labels, ks):
    #f1_weight = []
    #f1_macro = []
    #f1_micro = []
    f1 = []
    acc = []
    for k in ks:
        pred_k = stats.mode(knn_labels[:,:k], axis=1)[0].reshape((-1,))
        print(pred_k)
        #f1_weight.append(f1_score(test_labels, pred_k, average='weighted'))
        #f1_macro.append(f1_score(test_labels, pred_k, average='macro'))
        f1.append(f1_score(test_labels, pred_k))
        acc.append(accuracy_score(test_labels, pred_k))
    return {'f1': f1, 'accuracy': acc, }

def plot_prediction_quality(scores, ks):
    colors = ['r-', 'b-', 'g-','y-'][:len(scores)]
    for (k,v), color in zip(scores.items(), colors):
        plt.plot(ks, v, color, label=k)
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('prediction quality')
    plt.show()

def evaluate_quality(predictor, test_features, test_labels, model_name, verbose=True, num_batches=1):
    """
    Evaluate quality metrics of a model on a test set. 
    """
    # tune the predictor to provide the verbose response
    predictor.accept = 'application/json; verbose=true'
    
    # split the test data set into num_batches batches and evaluate using prediction endpoint. 
    print('running prediction (quality)...')
    batches = np.array_split(test_features, num_batches)
    knn_labels = []
    for batch in batches:
        pred_result = predictor.predict(batch)
        cur_knn_labels = np.array([pred_result['predictions'][i]['labels'] for i in range(len(pred_result['predictions']))])
        knn_labels.append(cur_knn_labels)
    knn_labels = np.concatenate(knn_labels)
    print('running prediction (quality)... done')
    print(knn_labels)
    print(test_labels)
    # figure out different k values
    top_k = knn_labels.shape[1]
    ks = range(1, top_k+1)
    
    # compute scores for the quality of the model for each value of k
    print('computing scores for all values of k... ')
    quality_scores = scores_for_ks(test_labels, knn_labels, ks)
    print('computing scores for all values of k... done')
    if verbose:
        plot_prediction_quality(quality_scores, ks)
    
    return quality_scores

def evaluate_latency(predictor, test_features, test_labels, model_name, verbose=True, num_batches=1):
    """
    Evaluate the run-time of a model on a test set.
    """
    # tune the predictor to provide the non-verbose response
    predictor.accept = 'application/json'
    
    # latency for large batches:
    # split the test data set into num_batches batches and evaluate the latencies of the calls to endpoint. 
    print('running prediction (latency)...')
    batches = np.array_split(test_features, num_batches)
    test_preds = []
    latency_sum = 0
    for batch in batches:
        start = time.time()
        pred_batch = predictor.predict(batch)
        latency_sum += time.time() - start
    latency_mean = latency_sum / float(num_batches)
    avg_batch_size = test_features.shape[0] / num_batches
    
    # estimate the latency for a batch of size 1
    latencies = []
    attempts = 128
    for i in range(attempts):
        start = time.time()
        pred_batch = predictor.predict(test_features[i].reshape((1,-1)))
        latencies.append(time.time() - start)

    latencies = sorted(latencies)
    latency1_mean = sum(latencies) / float(attempts)
    latency1_p90 = latencies[int(attempts*0.9)]
    latency1_p99 = latencies[int(attempts*0.99)]
    print('running prediction (latency)... done')
    
    if verbose:
        print("{:<11} {:.3f}".format('Latency (ms, batch size %d):' % avg_batch_size, latency_mean * 1000))
        print("{:<11} {:.3f}".format('Latency (ms) mean for single item:', latency1_mean * 1000))
        print("{:<11} {:.3f}".format('Latency (ms) p90 for single item:', latency1_p90 * 1000))
        print("{:<11} {:.3f}".format('Latency (ms) p99 for single item:', latency1_p99 * 1000))
        
    return {'Latency': latency_mean, 'Latency1_mean': latency1_mean, 'Latency1_p90': latency1_p90, 
            'Latency1_p99': latency1_p99}

def evaluate(predictor, test_features, test_labels, model_name, verbose=True, num_batches=100):
    eval_result_q = evaluate_quality(pred, test_features, test_labels, model_name=model_name, verbose=verbose, num_batches=num_batches)
    eval_result_l = evaluate_latency(pred, test_features, test_labels, model_name=model_name, verbose=verbose, num_batches=num_batches)
    return dict(list(eval_result_q.items()) + list(eval_result_l.items()))

def lassoSelection(X_train, y_train, n):
    '''
    Lasso feature selection.  Select n features. 
    '''
    #lasso feature selection
    #print (X_train)
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=0)
    sfm.fit(X_train, y_train)
    X_transform = sfm.transform(X_train)
    n_features = X_transform.shape[1]
    
    #print(n_features)
    while n_features > n:
        sfm.threshold += 0.01
        X_transform = sfm.transform(X_train)
        n_features = X_transform.shape[1]
    features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
    logger.info("selected features are {}".format(features))
    return features


def specificity_score(y_true, y_predict):
    '''
    true_negative rate
    '''
    true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
    real_negative = len(y_true) - sum(y_true)
    return true_negative / real_negative 

def delete_endpoint(predictor):
    try:
        boto3.client('sagemaker').delete_endpoint(EndpointName=predictor.endpoint)
        print('Deleted {}'.format(predictor.endpoint))
    except:
        print('Already deleted: {}'.format(predictor.endpoint))







# In[29]:


def trained_estimator_from_hyperparams(s3_train_data, hyperparams, output_path, s3_test_data=None):
    """
    Create an Estimator from the given hyperparams, fit to training data, 
    and return a deployed predictor
    
    """
    # specify algorithm containers. These contain the code for the training job
    containers = {
        'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/knn:1',
        'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/knn:1',
        'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/knn:1',
        'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/knn:1',
        'ap-northeast-1': '351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/knn:1',
        'ap-northeast-2': '835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/knn:1',
        'ap-southeast-2': '712309505854.dkr.ecr.ap-southeast-2.amazonaws.com/knn:1'
    }
    # set up the estimator
    knn = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
        get_execution_role(),
        train_instance_count=1,
        train_instance_type='ml.m5.2xlarge',
        output_path=output_path,
        sagemaker_session=sagemaker.Session())
    knn.set_hyperparameters(**hyperparams)
    
    # train a model. fit_input contains the locations of the train and test data
    fit_input = {'train': s3_train_data}
    if s3_test_data is not None:
        fit_input['test'] = s3_test_data
    knn.fit(fit_input)
    return knn


# In[30]:


def predictor_from_hyperparams(knn_estimator, estimator_name, instance_type, endpoint_name=None): 
    knn_predictor = knn_estimator.deploy(initial_instance_count=1, instance_type=instance_type,
                                        endpoint_name=endpoint_name)
    knn_predictor.content_type = 'text/csv'
    knn_predictor.serializer = csv_serializer
    knn_predictor.deserializer = json_deserializer
    return knn_predictor


# In[31]:


def highlight_apx_max(row):
    '''
    highlight the aproximate best (max or min) in a Series yellow.
    '''
    max_val = row.max()
    colors = ['background-color: yellow' if cur_val >= max_val * 0.9975 else '' for cur_val in row]
        
    return colors
def highlight_far_from_min(row):
    '''
    highlight the aproximate best (max or min) in a Series yellow.
    '''
    med_val = row.median()
    colors = ['background-color: red' if cur_val >= med_val * 1.2 else '' for cur_val in row]
        
    return colors


# In[ ]:



if __name__ == '__main__':


    data_dir ="data/"

    data_file = data_dir + "miRNA_matrix.csv"

    df = pd.read_csv(data_file)
    # print(df)
    y_data = df.pop('label').values

    df.pop('file_id')

    columns =df.columns
    #print (columns)
    X_data = df.values

    # split the data to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
    

    #standardize the data.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # check the distribution of tumor and normal sampels in traing and test data set.
    logger.info("Percentage of tumor cases in training set is {}".format(sum(y_train)/len(y_train)))
    logger.info("Percentage of tumor cases in test set is {}".format(sum(y_test)/len(y_test)))
    
    n = 7
    features_columns = lassoSelection(X_train, y_train, n)
    
    
    import io
    import sagemaker.amazon.common as smac
    import boto3
    import sagemaker  
    from sagemaker import get_execution_role
    from sagemaker.predictor import csv_serializer, json_deserializer
    bucket =  "cancer-bucket"
    prefix = 'knn-prediction'
    key = 'cancer-data'
    
    #write the train data to S3
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, X_train[:,features_columns], y_train)
    buf.seek(0)   
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
    s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)    
    print('uploaded training data location: {}'.format(s3_train_data)) 
    print(X_train[:,features_columns].shape)

    #write the test data to S3
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, X_test[:,features_columns], y_test)
    buf.seek(0)
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'test', key)).upload_fileobj(buf)
    s3_test_data = 's3://{}/{}/test/{}'.format(bucket, prefix, key)
    print('uploaded test data location: {}'.format(s3_test_data))    
    print(X_test[:,features_columns].shape)
    
    #scores = model_fit_predict(X_train[:,feaures_columns],X_test[:,feaures_columns],y_train,y_test)

    #draw(scores)
    #lasso cross validation
    # lassoreg = Lasso(random_state=0)
    # alphas = np.logspace(-4, -0.5, 30)
    # tuned_parameters = [{'alpha': alphas}]
    # n_fold = 10
    # clf = GridSearchCV(lassoreg,tuned_parameters,cv=10, refit = False)
    # clf.fit(X_train,y_train)


# In[18]:


hyperparams_flat_l2 = {
    'feature_dim': 5,
    'k': 100,
    'sample_size': 297,
    'predictor_type': 'classifier' 
    # NOTE: The default distance is L2 and index is Flat, so we don't list them here
}
output_path_flat_l2 = 's3://' + bucket + '/' + prefix + '/flat_l2/output'
knn_estimator_flat_l2 = trained_estimator_from_hyperparams(s3_train_data, hyperparams_flat_l2, output_path_flat_l2, 
                                                           s3_test_data=s3_test_data)


# In[ ]:


import time

instance_types = ['ml.m4.xlarge']
index2estimator = {'flat_l2': knn_estimator_flat_l2}

eval_results = {}

for index in index2estimator:
    estimator = index2estimator[index]
    eval_results[index] = {}
    for instance_type in instance_types:
        model_name = 'knn_%s_%s'%(index, instance_type)
        endpoint_name = 'knn-latency-%s-%s-%s'%(index.replace('_','-'), instance_type.replace('.','-'),
                                               str(time.time()).replace('.','-'))
        print('\nsetting up endpoint for instance_type=%s, index_type=%s' %(instance_type, index))
        pred = predictor_from_hyperparams(estimator, index, instance_type, endpoint_name=endpoint_name)
        print('')
        eval_result = evaluate(pred,X_test[:,features_columns], y_test, model_name=model_name, verbose=True)        
        eval_result['instance'] = instance_type 
        eval_result['index'] = index 
        eval_results[index][instance_type] = eval_result
        delete_endpoint(pred)


# In[74]:


import pandas as pd

k_range = range(1, 13)
df_index = []
data = []
columns_lat = ['latency_mean', 'latency1_mean', 'latency1_p90', 'latency1_p99']
columns_acc = ['acc_%d' % k for k in k_range]
columns = columns_lat + columns_acc
#print(eval_result)
print (eval_results)
for index, index_res in eval_results.items():
    print (index)
    print (index_res)
    for instance, res in index_res.items():
        # for sample size?
        print(instance)
        print(res)
        df_index.append(index+'_'+instance)
        latencies = np.array([res['Latency'], res['Latency1_mean'], res['Latency1_p90'], res['Latency1_p99']])
        row = np.concatenate([latencies*10,
                             res['accuracy'][k_range[0] - 1:k_range[-1] ]])
        row *= 100
        data.append(row)

df = pd.DataFrame(index=df_index, data=data, columns=columns)
df_acc = df[columns_acc]
df_lat = df[columns_lat]


df_acc.round(decimals=1).style.apply(highlight_apx_max, axis=1)
    


# In[75]:


df_lat.round(decimals=1).style.apply(highlight_far_from_min, axis=0)

