'''
d = {'banana': 3, 'apple': 4, 'pear': 1, 'orange': 2}
import collections

print(sorted(d.items()))

d['hell'] = 6

print(sorted(d.items()))
d.keys()
print (d.items())

class horse:
    def __init__(self,color,size):
        self.color=color
        self.size = size

    def __str__(self):
        print ('str')
        return '__str__ for horse'

    def __repr__(self):
        print ('repr')
        return '__repr__ for horse'


myhorse = horse("red","big")
myhorse
myhorse

print(myhorse)

print(myhorse.color)

repr(myhorse)

print(repr(myhorse))
str(myhorse)

import tensorflow as tf
import tokenization

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_file",'sample_text.txt',
                    "Input raw text file (or comma-separated list of files).")
input_files = []
for input_pattern in FLAGS.input_file.split(","):
    print (input_pattern)
    input_files.extend(tf.gfile.Glob(input_pattern))
for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
          line = reader.readline()
          if not line:
              break
          print(line)
'''



class statvdyna ():
    def __init__(self,x1,x2):
        self.x1 = x1
        self.x2 = x2
        print ('did __init__')
        print ('x1 + x2 is ', x1 + x2)

    @staticmethod
    def mystatic (ms1):
        print ('mystatic','  passed varoabl', ms1)
        #print(self.x1)
        #print(self.x2)

    @classmethod
    def mydyn(cls,ms2):
        print('mymethod', '  passed varoabl', ms2)




myclass = statvdyna (1,5)
print ('==========================')

myclass.mystatic(33)
print(myclass.mystatic(33))
print(type(myclass.mystatic(33)))
print ('==========================')

statvdyna.mystatic(33)
print(statvdyna.mystatic(33))
print(type(statvdyna.mystatic(33)))
print ('==========================')


myclass.mydyn (3333)
print(myclass.mydyn (3333))
print(type(myclass.mydyn (3333)))
print ('==========================')

statvdyna.mydyn(33)
print(statvdyna.mydyn(33))
print(type(statvdyna.mydyn(33)))
print ('==========================')


