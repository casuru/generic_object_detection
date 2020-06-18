import tensorflow as tf
from google.cloud import storage


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("BUCKET_NAME", None, "The name of google storage bucket.")
flags.DEFINE_string("STORAGE_FILE_PATH", None, "The path to the file you want to download.")
flags.DEFINE_string("LOCAL_FILE_PATH", None, "The path to where you want the file to download.")

flags.mark_flag_as_required("BUCKET_NAME")
flags.mark_flag_as_required("STORAGE_FILE_PATH")
flags.mark_flag_as_required("LOCAL_FILE_PATH")

def download_item_from_bucket(bucket_name, storage_file_path, local_file_path):

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(storage_file_path)
    blob.download_to_filename(local_file_path)

    return True

def main(_):

    download_item_from_bucket(FLAGS.BUCKET_NAME, FLAGS.STORAGE_FILE_PATH, FLAGS.LOCAL_FILE_PATH)


if __name__ == "__main__":

    tf.app.run()