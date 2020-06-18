import tensorflow as tf
import requests
import os
import json
from io import BytesIO
from PIL import Image
from object_detection.utils import dataset_util

ANAX_ROOT = "http://127.0.0.1:8000/"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("ANNOTATION_JSON_INPUT_PATH", None, "The json file containing the image annotations.")
flags.DEFINE_float("TRAINING_EVAL_SPLIT_PERCENTAGE", 0.8, "The percentage of the dataset to use for training.")
flags.DEFINE_string("TRAINING_OR_VALIDATION", "training", "Create the training or validation .tfrecord")
flags.DEFINE_string("TF_RECORD_OUTPUT_PATH", None, "The path to the .tfrecord file")
flags.mark_flag_as_required('ANNOTATION_JSON_INPUT_PATH')


def generate_tf_example(src, annotations):

    response = requests.get(src)
    img = Image.open(BytesIO(response.content))

    width = img.size[0]
    height = img.size[1]

    filename = b''

    encoded_image_data = img.tobytes()
    image_format = b"jpeg" if src.endswith(".jpg") or src.endswith(".jpeg") else b"png"

    xmins = [annotation["left"] for annotation in annotations]
    xmaxs = [annotation["left"] + annotation["width"] for annotation in annotations]
    ymins = [annotation["top"] for annotation in annotations]
    ymaxs = [annotation["top"] + annotation["height"] for annotation in annotations]
    classes = [annotation["class"] for annotation in annotations]
    classes_text = [str.encode(annotation["class_text"]) for annotation in annotations]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example



def main(_):

    with open(FLAGS.ANNOTATION_JSON_INPUT_PATH) as jf:

        annotations = json.load(jf)

        cut_off = int(len(annotations) * FLAGS.TRAINING_EVAL_SPLIT_PERCENTAGE)


        writer = tf.io.TFRecordWriter(FLAGS.TF_RECORD_OUTPUT_PATH)

        if FLAGS.TRAINING_OR_VALIDATION == "training":


            for annotation in annotations[: cut_off]:

                for src in annotation:

                    tf_example = generate_tf_example(src, annotation[src])
                    writer.write(tf_example.SerializeToString())

        if FLAGS.TRAINING_OR_VALIDATION == "validation":

            for annotation in annotations[cut_off:]:

                for src in annotation:

                    tf_example = generate_tf_example(src, annotation[src])
                    writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    
    tf.app.run()