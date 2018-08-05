# -*- coding: utf-8 -*-

from time import time
from datetime import datetime
import os
import csv
import numpy as np
import dill
from Tools.Printing import matrix_string

def classify(classifier, train_data=None, train_lbls=None, test_data=None, test_lbls=None, features=None, save=False, stats=False, save_dir='Results'):
    """"
    Wrapper function to perform training and classification and calculate statistics for analysis
    :param
        @classifier: classification object
        @train_data: training Data Set
        @test_data: testing Data Set
        @train_lbls: training Labels
        @features: list of feature columns to use for training and classification
        @save: store data and results in file
        @stats: save timing and training error
    :returns
        classifier: classifier object
        test_prediction: classification of test data
        test_score: accuracy of test prediction
        test_err_count: number of errors in test_prediction
    """
    if not features is None:
        train_data = train_data[:,features]
        test_data = test_data[:,features]

    # Training classifier
    if train_data is not None and train_lbls is not None:
        if stats:
            t0 = time()
        classifier.fit(train_data, train_lbls.ravel())
        if stats:
            train_time = time() - t0
            print('\tTraining finished in %0.3fs' % train_time,flush=True)

        # Classify training data (training error)
        if stats:
            train_prediction = classifier.predict(train_data)
            train_score, train_err_count, train_confusion_matrix = classifier.accuracy(train_prediction, train_lbls)

    # Classify test data
    if stats:
        t0 = time()
    test_prediction = classifier.predict(test_data)
    if stats:
        pred_time = time() - t0
        print('\tClassification finished in %0.3fs' % pred_time,flush=True)
    test_score, test_err_count, confusion_matrix = classifier.accuracy(test_prediction, test_lbls)

    # If specified, create directory for storing training and test data as well as results
    if save:
        # Create root results directory if does not exist
        root_results_dir = save_dir
        if not os.path.isdir(root_results_dir):
            os.makedirs(root_results_dir)

        # Create result directory for this classification
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        results_dir = root_results_dir + '/' + classifier.name + '_' + now
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        
        # Store training data and labels
        if train_data is not None and train_lbls is not None:
            if len(train_lbls.shape)==1:
                train_lbls = np.expand_dims(train_lbls,1)
            labeled_train_data = np.concatenate((train_data, train_lbls), axis=1)
            with open(results_dir + '/' + 'training_data.txt','w') as file:
                writer = csv.writer(file, delimiter=' ')
                for sample in labeled_train_data:
                    writer.writerow(sample)
        
        # Store training classification
        if stats:
            np.savetxt(results_dir + '/' + 'training_classification.txt', test_prediction,fmt='%i', delimiter="")
        
        # Store test data and labels
        if test_data is not None and test_lbls is not None:
            if len(test_lbls.shape)==1:
                test_lbls = np.expand_dims(test_lbls,1)
            labeled_test_data = np.concatenate((test_data, test_lbls), axis=1)
            with open(results_dir + '/' + 'test_data.txt','w') as file:
                writer = csv.writer(file, delimiter=' ')
                for sample in labeled_test_data:
                    writer.writerow(sample)
        
        # Store test classification
        np.savetxt(results_dir + '/' + 'test_classification.txt', test_prediction,fmt='%i', delimiter="")

        # Store results
        with open(results_dir + '/' + 'summary.txt','w') as file:
            if stats:
                file.write('Training time: '+ str(train_time)+'\n')
                file.write('Classification time: ' + str(pred_time)+'\n')
                file.write('Training score: ' + str(train_score)+'\n')
                file.write('Training error count: ' + str(train_err_count)+'\n')
                file.write('Training confusion matrix: '+matrix_string(train_confusion_matrix))
            file.write('Test score:' + str(test_score)+'\n')
            file.write('Test error count: ' + str(test_err_count)+'\n')
            file.write('Test confusion matrix: '+ matrix_string(confusion_matrix))
            if not features is None:
                file.write('Features: ' + ' '.join(str(e) for e in features))
        
        # Store the classifier object
        with open(results_dir+'/'+classifier.name+'.pkl','wb') as file:
            dill.dump(classifier,file)

    return classifier, test_prediction, test_score, test_err_count, confusion_matrix