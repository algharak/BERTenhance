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
import os
import os.path
import random

BOOK_SOURCE_DIR         =   'Training_Data_Raw_Text/txt'
BOOK_SOURCE_DIR_SHORT   =   'Training_Data_Raw_Text/txt_short'
BOOK_NUMBS              =   7
BOOK_NUMBS_SMALL        =   5
VOCAB_FILE_NAME         =   'Data/Vocab/vocab.txt'
TR_INSTANES_FILE        =   'output_file_tr.txt'
EV_INSTANCES_FILE       =   'output_file_ev.txt'
TR_EV_SPLIT             =   0.3
INSTANCES_DIR           =   'Instances'
BERT_CONFIG_FILE        =   'Data/Config/bert_config.json'
TF_CHECKPOINT_DIR       =   'Checkpoints'
NUM_INSERTION           =    5
PATH                    =   'Data'


scenario ={
            'do_train':             True,
            'do_eval':              False,
            'force_pretrn_gen_trn': True,
            'force_pretrn_gen_eval':True,
            'use_small_dataset':    False,
            'enable_enhancement':   False,
            'use_short_files':      False,
            'enable_original':      not 'enable_enhancement',
            'num_of_books_long':    BOOK_NUMBS,
            'num_of_books_short':   BOOK_NUMBS_SMALL,
            'enable_insertions':    True,
            'num_insertions':       NUM_INSERTION,
            'input_data_dir_long':  os.path.join(PATH,BOOK_SOURCE_DIR),
            'input_data_dir_short': os.path.join(PATH,BOOK_SOURCE_DIR_SHORT),
            'train_ev_split':       TR_EV_SPLIT,
            'vocab_file_path':      os.path.join(VOCAB_FILE_NAME),
            'tr_instances_file':    os.path.join(PATH,INSTANCES_DIR,TR_INSTANES_FILE),
            'ev_instances_file':    os.path.join(PATH,INSTANCES_DIR,EV_INSTANCES_FILE),
            'instances_dir':        os.path.join(PATH,INSTANCES_DIR),
            'bert_config_file':     os.path.join(BERT_CONFIG_FILE),
            'tf_chpt_dir':          os.path.join(PATH,TF_CHECKPOINT_DIR)
            }


def gen_list_dirfiles(dirname,booknumbs,splt):
    for _, _, files in os.walk(dirname):
        files = [files[i] for i in range(len(files)) if files[i][0] != '.'] #exclude files starting with .
        files = random.sample(files,booknumbs)
        #for i in files:
            #print (os.path.getsize(i))
        split_idx = round(splt*len(files))
        ev_file_list = files[:split_idx]
        tr_file_list = files [split_idx:]
        ev_files_str = [dirname + '/' + ev_file_list[i] for i in range(len(ev_file_list))]
        tr_files_str = [dirname + '/' + tr_file_list[i] for i in range(len(tr_file_list))]
        return ','.join(tr_files_str) , ','.join(ev_files_str)

def run_create_pretraining_data(**args):
    from create_pretraining_data import FLAGS as CFLAGS
    from create_pretraining_data import main as cr_pretrain_main
    if args['use_short_files']:
        gut_tr_eval_data_path = args ['input_data_dir_short']
        no_of_books = args['num_of_books_short']
    else:
        gut_tr_eval_data_path = args['input_data_dir_long']
        if args['use_small_dataset']:   no_of_books=args['num_of_books_short']
        else:                           no_of_books=args['num_of_books_long']
    tr_eval_split   =   args['train_ev_split']
    tr_instances,ev_instances = gen_list_dirfiles(gut_tr_eval_data_path,
                                                  no_of_books,
                                                  tr_eval_split)
    CFLAGS.vocab_file = args['vocab_file_path']
    # generate instances for training
    if args['force_pretrn_gen_trn']:
        if os.path.exists(args['tr_instances_file']):
            os.remove(args['tr_instances_file'])
        CFLAGS.enable_insertions = args['enable_insertions']
        CFLAGS.output_file = args['tr_instances_file']
        CFLAGS.input_file = tr_instances
        CFLAGS.num_of_insertions = args['num_insertions']
        cr_pretrain_main(1)

    if args['force_pretrn_gen_eval']:
        if os.path.exists(args['ev_instances_file']):
            os.remove(args['ev_instances_file'])
        CFLAGS.enable_insertions = False
        CFLAGS.output_file = args['ev_instances_file']
        CFLAGS.input_file = ev_instances
        cr_pretrain_main(1)



def run_training(**t_params):
    from create_pretraining_data import FLAGS as CFLAGS
    del_all_flags(CFLAGS) # this was done to avoid collision in the FLAGS over input_file
    from run_pretraining import FLAGS as RFLAGS
    from run_pretraining import main as rp_main
    RFLAGS.do_train = True
    #RFLAGS.do_eval = not RFLAGS.do_train  #this was a bug
    RFLAGS.bert_config_file = t_params['bert_config_file']
    RFLAGS.output_dir = t_params ['tf_chpt_dir']
    RFLAGS.input_file = t_params ['tr_instances_file']
    #the original size in run_pretraining module is 100,000.  Reducing for speedup
    RFLAGS.num_of_insertions = t_params['num_insertions']
    RFLAGS.enable_insertions = t_params['enable_insertions']

    RFLAGS.num_train_steps = 4000
    RFLAGS.num_warmup_steps = 2000 # added this since the loss drops faster
    RFLAGS.learning_rate = 5e-4 # the original was 5e-5
    rp_main(1)

def run_eval(**e_params):
    from create_pretraining_data import FLAGS as CFLAGS
    del_all_flags(CFLAGS) # this was done to avoid collision in the FLAGS over input_file
    from run_pretraining import FLAGS as RFLAGS
    from run_pretraining import main as rp_main
    RFLAGS.do_eval = e_params['do_eval']
    RFLAGS.bert_config_file = e_params['bert_config_file']
    RFLAGS.output_dir = e_params ['tf_chpt_dir']
    RFLAGS.input_file = e_params ['ev_instances_file']
    RFLAGS.max_eval_steps = 5000    #changed from 2000
    rp_main(1)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def main(**params):
    if params['force_pretrn_gen_eval'] or params['force_pretrn_gen_trn']:
        run_create_pretraining_data(**params)
    do_train = params['do_train']
    do_eval = params['do_eval']
    if do_train:
        run_training (**params)
    if do_eval:
        run_eval (**params)
    print("all done")
    return

main(**scenario)
exit()


if __name__ == "__main__":
    main()


