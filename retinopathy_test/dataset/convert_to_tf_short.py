from random import shuffle
import glob, os, sys, cv2
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd

IMG_SIZE = 256

def load_image(addr, img_size):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #COLOR_BGR2RGB

    img = img.astype(np.float32)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

parser = argparse.ArgumentParser()
parser.add_argument('--show', default=False,
                    help="Directory containing the dataset")

if __name__ == '__main__':
    args = parser.parse_args()
    show = args.show

    for s in ['tr_short','va_short']:

        # Divide the hata into 60% train, 20% validation, and 20% test
        dataset = 'records/retinopathy'
        name = dataset
        data = pd.read_csv('records/'+s+'.lst',header=None, sep='\t')
        print(data)
        addrs = data[2]
        labels = data[1]

        train_filename = name+'_'+s+'.tfrecords'  # address to save the TFRecords file
        print('generating', train_filename)
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(train_filename)
        for i in range(len(addrs)):
            # print how many images are saved every 1000 images
            if not i % 100: #ki: changed from 1000 to 100
                print('Train data: {}/{}'.format(i, len(addrs)))
                sys.stdout.flush()
            # Load the image
            img = load_image(addrs[i], IMG_SIZE)
            label = labels[i]
            # Create a feature
            feature = {'label': _int64_feature(label),
                    'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            
        writer.close()
        sys.stdout.flush()
