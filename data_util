# encoding: utf-8
# file: data_util.py
# author: shawn233

from __future__ import print_function
import os
import sys
import time
import numpy as np

'''
Functions:
1. divide data into training set, dev set and test set (7:1:2)
2. provide function `next_batch()`, returns the next batch in one epoch;
   provide function `reset_batch()`, to reset the batch for a new epoch.

Usage tips:
1. Assume memory is large enough to store all data, will use `readlines()` to read data;
'''

'''
TODO:
1. divide data based on date; (done)
2. add time stamp to feature; (done)
3. delete midprice for feature; (done)
4. clean data to include only 3-sec intervals; (done)
5. remove time stamp intervals from features; (done)
6. attemp: remove volume from features; success? (seens not)
7. fine processing of data_matrix to get more train data: 
   eliminate intervals less than 3 secs; (done) (may need to discard)
8. correct the mistake of crossing a day when calculating mean mid price;
9. correct the mistake of deleting time stamps for calculating mean mid price;

'''

TRAIN_INPUTS_FILENAME = 'train_inputs.npy'
TRAIN_LABELS_FILENAME = 'train_labels.npy'
DEV_INPUTS_FILENAME = 'dev_inputs.npy'
DEV_LABELS_FILENAME = 'dev_labels.npy'
TEST_INPUTS_FILENAME = 'test_inputs.npy'
TEST_LABELS_FILENAME = 'test_labels.npy'

TRAIN_MEANS_FILENAME = 'train_means.npy'
TRAIN_STDDEVS_FILENAME = 'train_stddevs.npy'
DEV_MEANS_FILENAME = 'dev_means.npy'
DEV_STDDEVS_FILENAME = 'dev_stddevs.npy'
TEST_MEANS_FILENAME = 'test_means.npy'
TEST_STDDEVS_FILENAME = 'test_stddevs.npy'

TRAIN_DATA_FILENAME = 'train_data.csv'
TEST_DATA_FILENAME = 'test_data.csv'

TRAIN_DATA_PRE_PROCESSED_FILENAME = 'train_preprocessed.npy'
TEST_DATA_PRE_PROCESSED_FILENAME = 'test_preprocessed.npy'
PRE_PROCESS_RECORD_FILENAME = 'preprocess.txt'


def _save_data (inputs, labels, full_path_dir, inputs_name, label_name):
    '''
    Save data into full_path_dir
    
    used in function `divide_data()`
    '''

    #arr_inputs = np.array(inputs, dtype=np.float32)
    #arr_labels = np.array(labels, dtype=np.float32)

    np.save(os.path.join (full_path_dir, inputs_name), inputs)
    np.save(os.path.join (full_path_dir, label_name), labels)


def _read_data (full_path_dir, inputs_name, labels_name):
    '''
    Read data from full_path_dir
    
    Returns:
        inputs, labels
    '''

    return np.load (os.path.join (full_path_dir, inputs_name)),\
            np.load (os.path.join (full_path_dir, labels_name))


class OrderBook:

    '''
    Order book class, designed mainly for data input
    '''

    def __init__ (self, batch_size, data_dir, 
        num_inputs=10,\
        num_labels=20,\
        data_regenerate_flag=False):
        '''
        Initialization, open the files and set the arguments

        Args:
        - batch_size: int;
        - data_dir: string, directory of the data
        - data_regenerate_flag: bool, True if re-process data, False if use stored data
        '''

        self._batch_size = batch_size
        self.batch_ind = 0
        self._data_dir = data_dir
        self._num_inputs = num_inputs
        self._num_labels = num_labels
        self.num_features = None # will be later set after processing data

        # vars for training set
        self.train_inputs = None
        self.train_labels = None
        self.train_means = None
        self.train_stddevs = None

        # vars for dev set
        self.dev_inputs = None
        self.dev_labels = None
        self.dev_means = None
        self.dev_stddevs = None

        # vars for test set
        self.test_inputs = None
        self.test_labels = None
        self.test_means = None
        self.test_stddevs = None

        # var for recording index in data matrix
        self.index = {
            'Date':1,
            'Time':2,
            'MidPrice':3,
            'LastPrice':4,
            'Volume':5,
            'BidPrice1':6,
            'BidVolume1':7,
            'AskPrice1':8,
            'AskVolume1':9,
            'TimeStamp':10
        }
        
        if data_regenerate_flag or not os.path.exists (os.path.join (self._data_dir, TRAIN_INPUTS_FILENAME)):
            self.__data_process_procedure()

        self.train_inputs, self.train_labels, self.train_means, self.train_stddevs = \
                self.__load_inputs_and_labels(os.path.join (self._data_dir, TRAIN_INPUTS_FILENAME),\
                                              os.path.join (self._data_dir, TRAIN_LABELS_FILENAME),\
                                              os.path.join (self._data_dir, TRAIN_MEANS_FILENAME),\
                                              os.path.join (self._data_dir, TRAIN_STDDEVS_FILENAME))

        self.num_features = self.train_inputs.shape[2]




    @property
    def batch_size (self):
        return self._batch_size

    
    @batch_size.setter
    def batch_size (self, value):
        self._batch_size = value

    
    @property
    def data_dir (self):
        return self._data_dir


    @data_dir.setter
    def data_dir (self, value):
        self._data_dir = value


    @property
    def num_samples (self):
        '''
        Number of training samples
        '''

        return self.train_inputs.shape[0]

    
    @property
    def num_batches (self):
        '''
        Maximum number of batches that can be provided in one epoch
        '''
        return int (self.num_samples / self.batch_size)



    def __data_process_procedure (self):
        '''
        Define the procedure of data processing

        Args:
        None

        Returns:
        None
        '''

        # train data
        print ("Start processing training data")
        print ("Reading data matrix...")
        data_matrix = \
                    self.__read_data_matrix (os.path.join (self._data_dir, TRAIN_DATA_FILENAME))
        print ("Done")
        print ("Dividing data matrix into days...")
        day_matrix_list = \
                    self.__divide_by_day (data_matrix)
        print ("Done")
        print ("Generating samples...")
        sample_inputs_list, sample_labels_list, base_index = \
                    self.__generate_samples (day_matrix_list)
        print ("Done")
        print ("Normalizing samples...")
        sample_inputs_list, sample_labels_list, mean_list, stddev_list = \
                    self.__sample_normalization (sample_inputs_list, sample_labels_list, base_index)
        print("Done")
        print("Remove lastPrice feature...")
        sample_inputs_list, sample_labels_list, mean_list, stddev_list = \
            self.__remove_lastPrice(sample_inputs_list, sample_labels_list, base_index)
        print ("Done")
        print ("Saving samples...")
        train_inputs_path, train_labels_path, train_means_path, train_stddevs_path = \
                    self.__store_inputs_and_labels (sample_inputs_list, sample_labels_list, mean_list, stddev_list)
        print ("Done")
        print ("Processing training data completed")

        # test data
        print ("Start procssing test data")
        print ("Reading data matrix...")
        data_matrix = \
                    self.__read_data_matrix (os.path.join (self._data_dir, TEST_DATA_FILENAME))
        print ("Done")
        print ("Parsing test data...")
        test_inputs_list, base_index = \
                    self.__parse_test_data (data_matrix)
        print ("Done")
        print ("Normalizing test inputs...")
        meaningless_test_labels_list = np.zeros (shape=[len(test_inputs_list)]) # just fit the arguments of __sample_normalization
        test_inputs_list, meaningless_test_labels_list, mean_list, stddev_list = \
                    self.__sample_normalization (test_inputs_list, meaningless_test_labels_list, base_index)
        print ("Done")
        print ("Saving test inputs...")
        self.__store_test_inputs (test_inputs_list, mean_list, stddev_list)
        print ("Done")
        print ("Procssing test data completed")



    def __read_data_matrix(self, in_filename):
        '''
        Read the train data matrix

        Args:
        - in_filename: string, input file name;

        Returns:
        - data_matrix: 2-d np matrix, dtype=<U32; 
        '''
        in_f = open (in_filename, 'r')

        data_matrix = []

        in_f.readline()
        for raw_line in in_f:
            # jump through empty lines
            line = raw_line.strip()
            if line == '':
                continue
            # process csv line
            line = line.split (',')
            refer1 = self.__get_time_stamp(line[self.index['Date']], '09:30:00')
            refer2 = self.__get_time_stamp(line[self.index['Date']], '11:30:00')
            refer3 = self.__get_time_stamp(line[self.index['Date']], '13:00:00')
            refer4 = self.__get_time_stamp(line[self.index['Date']], '15:00:00')
            timeStamp = self.__get_time_stamp(line[self.index['Date']], line[self.index['Time']]) # replace date and time with time stamp
            if timeStamp < refer1 or refer2 < timeStamp < refer3 or timeStamp > refer4:
                continue  # remove the data not in transaction time
            line.append (timeStamp)
            data_matrix.append (line)
        
        in_f.close()

        data_matrix = np.asarray (data_matrix)
        print ('data matrix shape:', data_matrix.shape)
        
        return data_matrix



    def __divide_by_day (self, input_matrix):
        '''
        Divide train data by day and morning/afternoon

        Args:
        - input_matrix: 2-d np matrix, dtype=<U32;

        Returns:
        - day_matrix_list: a list of 2-d np matrix;
        '''
        
        prev_date = None
        morning = False
        day_split_pos_list = [] # records the index of the first row in a new day

        for i in range (input_matrix.shape[0]):
            if prev_date is None or prev_date != input_matrix[i, self.index['Date']]:
                # a new day
                line = input_matrix[i]
                refer1 = self.__get_time_stamp(line[self.index['Date']], '09:30:00')
                refer2 = self.__get_time_stamp(line[self.index['Date']], '11:30:00')
                refer3 = self.__get_time_stamp(line[self.index['Date']], '13:00:00')
                refer4 = self.__get_time_stamp(line[self.index['Date']], '15:00:00')
                if not morning and refer1<line[-1]<refer2:
                    morning = True
                    day_split_pos_list.append (i)
                if morning and refer3<line[-1]<refer4:
                    morning = False
                    day_split_pos_list.append (i)
                prev_date = input_matrix[i, self.index['Date']]

        print ('total number of days:', len (day_split_pos_list))

        day_matrix_list = []
        for i in range (len (day_split_pos_list)-1):
            day_matrix_list.append (input_matrix[day_split_pos_list[i]:day_split_pos_list[i+1], :])
        day_matrix_list.append (input_matrix[day_split_pos_list[-1]:, :])

        return day_matrix_list
        
        



    def __generate_samples (self, day_matrix_list):
        '''
        From the list of day matrix, generate samples

        Args:
        - day_matrix_list: a list of 2-d np matrix, dtype=<U32;

        Returns:
        - sample_inputs_list: a list of input samples (2-d np matrix);
        - sample_labels_list: a list of labels, which is a float numebr;
        - base_index: int, the first column of inputs is the base_index-th column in data_matrix;
        '''
        
        sample_inputs_list = []
        sample_labels_list = []

        base_index = self.index['MidPrice']

        for ind_day in range (len(day_matrix_list)):
            day_matrix = day_matrix_list[ind_day]
            num_samples = day_matrix.shape[0] - self._num_inputs - self._num_labels + 1 # yes it is
            
            for i in range (num_samples):
                new_input = day_matrix[i:(i+self._num_inputs), base_index:].astype (np.float32)
                new_label = np.mean (day_matrix[(i+self._num_inputs): (i+self._num_inputs+self._num_labels),\
                                     self.index['MidPrice']:self.index['MidPrice']+1].astype(np.float32))
                sample_inputs_list.append (new_input)
                sample_labels_list.append (new_label)

            # TODO: may need drop the first sample
            # why just mean of next samples

        print ("Total number of samples", len(sample_inputs_list))
        return sample_inputs_list, sample_labels_list, base_index





    def __sample_normalization (self, sample_inputs_list, sample_labels_list, base_index):
        '''
        Normalize the samples, formula:
        value = (value - mean) / stddev

        Args:
        - sample_inputs_list: a list of input samples (2-d np matrix);
        - sample_labels_list: a list of labels, which is a float numebr;
        - base_index: int, the first column of inputs is the base_index-th column in data_matrix;

        Returns:
        - sample_inputs_list: a list of normalized input samples (2-d np matrix);
        - sample_lables_list: a list of labels (float); label is
            (original label (mean mid price)- mean mid price in inputs) / stddev mid price in inputs;
        - mean_list: a list of float, mean value of mid price;
        - stddev_list: a list of float, stddev value of mid price;
        '''
        
        assert len (sample_inputs_list) == len (sample_labels_list)

        mean_list = []
        stddev_list = []


        for i in range (len (sample_inputs_list)):
            sample_inputs = sample_inputs_list[i]
            sample_labels = sample_labels_list[i]

            mean = np.mean (sample_inputs, axis=0)
            stddev = np.std (sample_inputs, axis=0)
            stddev = np.maximum (stddev, 1e-6) # to prevent zero division problems

            mean_list.append (mean[self.index['MidPrice'] - base_index])
            stddev_list.append (stddev[self.index['MidPrice'] - base_index])

            # normalize inputs, calculate new label
            sample_inputs = (sample_inputs - mean) / stddev
            sample_labels = sample_labels - mean_list[i]

            sample_inputs_list[i] = sample_inputs[:, (self.index['LastPrice'] - base_index):]
            sample_labels_list[i] = sample_labels

        return sample_inputs_list, sample_labels_list, mean_list, stddev_list



    def __remove_lastPrice (self, sample_inputs_list, sample_labels_list, mean_list, stddev_list):
        '''

        sample_inputs_list:param sample_inputs_list:
        :param sample_labels_list:
        :param mean_list:
        :param stddev_list:
        :return:
        '''
        sample_inputs_list = sample_inputs_list[:, 1:]
        return sample_inputs_list, sample_labels_list, mean_list, stddev_list



    def __store_inputs_and_labels (self, sample_inputs_list, sample_labels_list, mean_list, stddev_list):
        '''
        Store sample inputs and labels as np arrays

        Args:
        - sample_inputs_list: a list of inputs;
        - sample_lables_list: a list of labels;

        Returns:
        inputs_file_path: full path of inputs file;
        labels_file_path: full path of labels file;
        mean_file_path: full path of mean value file;
        stddev_file_path: full path of stddev value file;
        '''

        sample_inputs_list = np.asarray (sample_inputs_list, dtype=np.float32)
        sample_labels_list = np.asarray (sample_labels_list, dtype=np.float32).reshape([-1, 1])
        mean_list = np.asarray (mean_list, dtype=np.float32)
        stddev_list = np.asarray (stddev_list, dtype=np.float32)

        print ("sample inputs shape", sample_inputs_list.shape)
        print ("sample labels shape", sample_labels_list.shape)
        print ("mean list shape", mean_list.shape)
        print ("stddev list shape", stddev_list.shape)
        self.num_features = sample_inputs_list.shape[2]

        inputs_file_path = os.path.join (self._data_dir, TRAIN_INPUTS_FILENAME)
        labels_file_path = os.path.join (self._data_dir, TRAIN_LABELS_FILENAME)
        mean_file_path = os.path.join (self._data_dir, TRAIN_MEANS_FILENAME)
        stddev_file_path = os.path.join (self._data_dir, TRAIN_STDDEVS_FILENAME)
        
        np.save (inputs_file_path, sample_inputs_list)
        np.save (labels_file_path, sample_labels_list)
        np.save (mean_file_path, mean_list)
        np.save (stddev_file_path, stddev_list)

        return inputs_file_path,\
               labels_file_path,\
               mean_file_path,\
               stddev_file_path        




    def __load_inputs_and_labels (self, inputs_file_path, labels_file_path, mean_file_path, stddev_file_path):
        '''
        Load sample inputs and labels

        Args:
        - inputs_file_path: string;
        - labels_file_path: string;
        - mean_file_path: string;
        - stddev_file_path: string;

        Returns:
        - inputs: 3-d np array;
        - labels: 2-d np array;
        - means: 1-d np array;
        - stddevs: 1-d np array;
        '''

        return np.load (inputs_file_path),\
               np.load (labels_file_path),\
               np.load (mean_file_path),\
               np.load (stddev_file_path)




    def next_batch_with_mean_and_stddev (self):
        '''
        next_batch() interface for normalization dedicated for every input

        Args:
        None

        Returns:
        - inputs: a 3-d np array, batch_size x self._num_inputs x self._num_features;
        - labels: a 2-d np array, batch_size x 1;
        - means: a 1-d np array, batch_size
        - stddevs: a 1-d np array, batch_size
        '''

        assert self.batch_ind + self.batch_size <= self.train_inputs.shape[0]

        train_batch_inputs = self.train_inputs [self.batch_ind: self.batch_ind + self.batch_size, :, :]
        train_batch_labels = self.train_labels [self.batch_ind: self.batch_ind + self.batch_size, :]
        train_batch_means = self.train_means [self.batch_ind: self.batch_ind + self.batch_size]
        train_batch_stddevs = self.train_stddevs [self.batch_ind: self.batch_ind + self.batch_size]

        self.batch_ind += self.batch_size

        return train_batch_inputs,\
               train_batch_labels,\
               train_batch_means,\
               train_batch_stddevs



    def __parse_test_data (self, data_matrix):
        '''
        Parse test data into samples

        Args:
        - data_matrix: 2-d np matrix, dtype=<U32

        Returns:
        - test_inputs_list: a list of test inputs, which is a 2-d np matrix;
        - base_index: int, see __sample_normalization ()
        '''

        test_inputs_list = []
        num_inputs = data_matrix.shape[0] // self._num_inputs

        assert type (num_inputs) == type (1) # at least I think so
        assert num_inputs * self._num_inputs == data_matrix.shape[0]

        base_index = self.index['MidPrice']
        for i in range (num_inputs):
            test_inputs_list.append (data_matrix[i*self._num_inputs:(i+1)*self._num_inputs,\
                                                 base_index:].astype (np.float32))

        print ('Total number of test inputs', len (test_inputs_list))

        return test_inputs_list, base_index



    def __store_test_inputs (self, test_inputs_list, mean_list, stddev_list):
        '''
        Save test inputs data as np arrays

        Args:
        - test_inputs_list: a list of test inputs (2-d np array);
        - mean_list: 1-d np array;
        - stddev_list: 1-d np array;

        Returns:
        - test_inputs_path;
        - test_means_path;
        - test_stddevs_path;
        '''

        test_inputs_path = os.path.join (self._data_dir, TEST_INPUTS_FILENAME)
        test_means_path = os.path.join (self._data_dir, TEST_MEANS_FILENAME)
        test_stddevs_path = os.path.join (self._data_dir, TEST_STDDEVS_FILENAME)
        
        np.save (test_inputs_path, np.asarray (test_inputs_list))
        np.save (test_means_path, mean_list)
        np.save (test_stddevs_path, stddev_list)

        return test_inputs_path, test_means_path, test_stddevs_path


    def __load_test_inputs (self, test_inputs_path, test_means_path, test_stddevs_path):
        '''
        Load test inputs, means and stddevs

        Args:
        - test_inputs_path;
        - test_means_path;
        - test_stddevs_path;

        Returns:
        - test_inputs_list: 3-d np array;
        - test_means_list: 1-d np array;
        - test_stddevs_list: 1-d np array;
        '''

        return np.load (test_inputs_path), np.load (test_means_path), np.load (test_stddevs_path)



    def __get_time_stamp (self, date, acc_time):
        '''
        Get the time stamp from date and time

        Args:
        - date: string, form: %Y/%m/%d;
        - acc_time: string, form: %H:%M:%S;
        (for info of %Y, %m, %M, etc., see doc for time.strptime)

        Returns:
        - timestamp: int
        '''

        form = r'%Y-%m-%d %H:%M:%S'
        #print ('date', date)
        #print ('time', acc_time)
        time_array = time.strptime (date+' '+acc_time, form)
        time_stamp = int (time.mktime (time_array))
        #print ('timestamp', str(time_stamp))

        return time_stamp




    def __validate_input (self, input_matrix, midprice_matrix, show_error=False):
        '''
        [Discard]
        Check if an input matrix satisfies the requirements:
        - shape is: (self._num_inputs, self._num_features+1), +1 because time stamp in input_matrix now
        - time stamp interval must equal to 3.0

        Check if the mid price matrix satisfies the requirements:
        - shape is: (self._num_lables, 1);
        - can not cross a day;

        Args:
        - input_matrix: 2-d np array;
        - show_error: bool, True if show error info

        Returns:
        - boolean value
        '''
        if input_matrix.shape != (self._num_inputs, self._num_features+1):
            if show_error:
                print ('[validate error] input matrix shape error:', input_matrix.shape)
            return False

        if midprice_matrix.shape != (self._num_labels, 1):
            if show_error:
                print ('[validate error] mid price matrix shape error:', midprice_matrix.shape)
            return False
        
        for i in range (self._num_inputs):
            if np.abs(input_matrix[i, 0] - 3.0) > 1e-4:
                if show_error:
                    print ('[validate error] input matrix timestamp error:')
                    print (input_matrix)
                return False

        for i in range (self._num_labels):
            if np.abs (midprice_matrix[i, 0]) > 3.1:
                if show_error:
                    print ('[validate error] mid price matrix crosses a day')
                    print (midprice_matrix)
                return False

        return True



    def __divide_data (self, in_filename):
        '''
        [Discard]
        Divide data into training set, and dev set (9:1), **after pre-processing**
        
        Args:
        - in_filename: string, full path of pre-processed data file
        
        Returns:
        - None

        (Implementation specified for projects, can not be reused)
        '''
        
        input_size = 10
        output_avg_len = 20

        data_matrix = np.load (in_filename)
        
        # 1. generate inputs and lables from data_matrix
        inputs = []
        labels = []

        total_cnt = 0
        accepted_cnt = 0

        num_inputs = data_matrix.shape[0] - (input_size + output_avg_len) + 1

        # mean and stddev prepared for later calculation
        pre_mean_labels = self.pre_mean[1]
        pre_stddev_labels = self.pre_stddev[1]
        
        for i in range (num_inputs):
            # delete midprice from input features 
            input_matrix = np.hstack ((data_matrix[i:(i+input_size), :1], 
                                      data_matrix[i:(i+input_size), 2:]))
            total_cnt += 1
            
            input_matrix[0, 0] = 3.0
            midprice_matrix = data_matrix[i+input_size:i+input_size+output_avg_len, 1:2]
            midprice_timestamp_matrix = data_matrix[i+input_size:i+input_size+output_avg_len, :1]
            if (self.__validate_input (input_matrix, midprice_timestamp_matrix)):
                accepted_cnt += 1
                inputs.append (input_matrix[:, 1:])
                label_val = np.mean (midprice_matrix)
                labels.append ((label_val-pre_mean_labels)/pre_stddev_labels)

        assert len(inputs) == len(labels)
        print ('accepted train samples:', accepted_cnt, '/', total_cnt)

        # 2. divide train data and dev data
        indices = np.asarray(list (range(len(inputs))))
        np.random.shuffle (indices)

        train_data_bound = int (np.ceil(len(inputs) * 0.9))
        dev_data_bound = len (inputs)

        train_inputs = []
        train_labels = []
        dev_inputs = []
        dev_labels = []

        for i in indices[:train_data_bound]:
            train_inputs.append (inputs[i])
            train_labels.append (labels[i])

        for i in indices[train_data_bound:]:
            dev_inputs.append (inputs[i])
            dev_labels.append (labels[i])

        train_inputs = np.asarray (train_inputs)
        train_labels = np.asarray (train_labels)
        dev_inputs = np.asarray (dev_inputs)
        dev_labels = np.asarray (dev_labels)

        # 3. save train and dev data
        full_path_dir = os.path.dirname (in_filename)
        _save_data (train_inputs, train_labels, full_path_dir, TRAIN_INPUTS_FILENAME, TRAIN_LABELS_FILENAME)
        _save_data (dev_inputs, dev_labels, full_path_dir, DEV_INPUTS_FILENAME, DEV_LABELS_FILENAME)
        #_save_data (test_inputs, test_labels, full_path_dir, TEST_INPUTS_FILENAME, TEST_LABELS_FILENAME)



    
    def __read_test_data (self, in_filename):
        '''
        [Discard]
        Read and save test data set after pre-process test data

        Args:
        - in_filename: string, path of pre-processed test data.

        Returns:
        - None
        '''
    
        input_size = 10
        data_matrix = np.load (in_filename)

        num_inputs = data_matrix.shape[0] // input_size
        assert num_inputs * input_size == data_matrix.shape[0]

        inputs = []
        for i in range (num_inputs):
            input_matrix = np.hstack ((data_matrix[i*input_size:(i+1)*input_size, :1],
                                       data_matrix[i*input_size:(i+1)*input_size, 2:]))
            input_matrix[0, 0] = 3.0
            #assert self.__validate_input (input_matrix, show_error=True)
            inputs.append (input_matrix[:, 1:])

        full_path_dir = os.path.dirname (in_filename)
        _save_data (inputs, [], full_path_dir, TEST_INPUTS_FILENAME, TEST_LABELS_FILENAME)




    def next_batch (self):
        '''
        Get the next batch of the training set

        Returns:
            train_inputs_batch: a padded input batch, batch_size x max_len x n_input
            train_labels_batch: a padded label batch, batch_size x 1
        '''

        if self.train_inputs is None:
            self.train_inputs, self.train_labels = _read_data (self.data_dir, TRAIN_INPUTS_FILENAME, TRAIN_LABELS_FILENAME)

        assert self.batch_ind + self.batch_size <= len (self.train_inputs)

        train_batch_inputs = self.train_inputs[self.batch_ind: self.batch_ind + self.batch_size]
        train_batch_labels = self.train_labels[self.batch_ind: self.batch_ind + self.batch_size]

        self.batch_ind += self.batch_size

        return train_batch_inputs, train_batch_labels


    def reset_batch (self):
        '''
        Reset self.batch_ind for a new epoch
        '''

        self.batch_ind = 0


    def dev_set (self):    
        '''
        Get the padded dev inputs and labels

        Returns:
            dev_inputs: a list of inputs (lists);
            dev_lables: a list of labels
        '''

        if self.dev_inputs is None:
            self.dev_inputs, self.dev_labels = _read_data (self.data_dir, DEV_INPUTS_FILENAME, DEV_LABELS_FILENAME)

        return self.dev_inputs, self.dev_labels


    def test_set (self):
        '''
        Get the test inputs, means and stddevs

        Returns:
            test_inputs: 3-d np array;
            test_means: 1-d np array;
            test_stddevs: 1-d np array;
        '''

        if self.test_inputs is None:
            self.test_inputs, self.test_means, self.test_stddevs = \
                self.__load_test_inputs (os.path.join (self._data_dir, TEST_INPUTS_FILENAME),\
                                         os.path.join (self._data_dir, TEST_MEANS_FILENAME),\
                                         os.path.join (self._data_dir, TEST_STDDEVS_FILENAME))

        return self.test_inputs, self.test_means, self.test_stddevs





if __name__ == "__main__":
    BASE_DIR = os.path.dirname (os.path.abspath(sys.argv[0]))
    #INPUT_FILENAME = 'train1.csv'
    PROJECT_DIR = os.path.dirname (BASE_DIR)
    DATA_DIR = os.path.join (PROJECT_DIR, 'data')

    order_book = OrderBook (2, DATA_DIR, data_regenerate_flag=True)
    
    print (order_book.num_batches)
    for i in range (10):
        inputs, labels, mean, stddev = order_book.next_batch_with_mean_and_stddev ()
        print (inputs)
        print (labels)
        print (mean)
        print (stddev)
        input ()

    #test_inputs, _ = order_book.test_set()
    #print (test_inputs.shape)

