# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

import tensorflow as tf
import os.path
#from create_pretraining_data import FLAGS as CPD_FLAGS
#from create_pretraining_data import main as CPD_main
#from run_pretraining import main as RP_main
#from run_pretraining import FLAGS as RP_FLAGS

BOOK_SOURCE_DIR = '/Users/algh/als_code/bert/TrainingDataGutenberg/Gutenberg/BookRawText/txt_short'
VOCAB_FILE = '/Users/algh/als_code/bert/multilingual_L-12_H-768_A-12/vocab.txt'
TF_CHECKPOINT_FILE = '/Users/algh/als_code/bert/TrainingDataGutenberg/model_out_dir'
NUM_OF_BOOKS_TDS = 5    # to save on the run time
INSTANCES_FILE = '/Users/algh/als_code/bert/TrainingDataGutenberg/Training_Instances/output_file.txt'
INSTANCES_DIR = '/Users/algh/als_code/bert/TrainingDataGutenberg/Training_Instances'
BERT_CONFIG_FILE = '/Users/algh/als_code/bert/multilingual_L-12_H-768_A-12/bert_config.json'
TF_CHECKPOINT_FILE = '/Users/algh/als_code/bert/TrainingDataGutenberg/TF_checkpoints/output.txt'


def gen_list_dirfiles(dirname):
    for _, _, files in os.walk(dirname):
        files = [files[i] for i in range(len(files)) if files[i][0] != '.'] #exclude files starting with .
        files = [dirname + '/' + files[i] for i in range(len(files))]
        return ','.join(files)

def run_create_pretraining_data(gut_tr_data):
    from create_pretraining_data import FLAGS as CFLAGS
    from create_pretraining_data import main
    book_list = gen_list_dirfiles(gut_tr_data)
    CFLAGS.output_file = INSTANCES_FILE
    CFLAGS.input_file = book_list
    CFLAGS.vocab_file = VOCAB_FILE
    if not os.path.isfile(INSTANCES_FILE):
        main(1)

def run_runpretraining(instances):
    from create_pretraining_data import FLAGS as CFLAGS
    del_all_flags(CFLAGS)        # this was done to avoid collision in the FLAGS over input_file
    from run_pretraining import FLAGS as RFLAGS
    from run_pretraining import main
    RFLAGS.do_train = True
    RFLAGS.bert_config_file = BERT_CONFIG_FILE
    RFLAGS.output_dir = TF_CHECKPOINT_FILE
    RFLAGS.input_file = gen_list_dirfiles(instances)
    main(1)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

run_create_pretraining_data(BOOK_SOURCE_DIR)
run_runpretraining(INSTANCES_DIR)

#th.exists(TRAIN_DATA_FILE): run_create_pretraining_data(BOOK_SOURCE_DIR)

#run_run_pretraining(BOOK_SOURCE_DIR)







