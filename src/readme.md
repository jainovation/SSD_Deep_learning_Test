참고사이트 : http://www.birc.co.kr/2018/02/18/object-detection-with-tensorflow-api/


사용환경 : Linux, python3, tensorflow api
>### step1 : Tensorflow API 설치
 바탕화면에서 http://www.birc.co.kr/download/2809/ 에서 models파일을 다운받은 후 압축해제한다.
 터미널을 실행한 후 다음과 같은 라이브러리를 설치
 sudo pip install pillow   - (설치가 안될 경우 pip3로 수정해서 진행하고 pip가 업그레이드 오류 시 sudo python3 -m pip uninstall pip && sudo                              apt-get install python3-pip --reinstall)
 sudo pip install lxml
 sudo pip install jupyter
 sudo pip install matplotlib
 cd Desktop/models
 protoc object_detection/protos/*.proto --python_out=.
 export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
 sudo python3 setup.py install
 
>### step2 : 이미지 수집
traffic light 사진 수집

>### step3 : object labeling

git clone https://github.com/tzutalin/labelImg
sudo apt-get install pyqt5-dev-tools
cd labelImg
make qt5py3
python3 labelImg.py

>### step4 : TF-record 만들기

‘object-detection’을 바탕화면에서 생성 후 ‘object-detection’폴더에서 새로운 문서의 이름을 ‘xml_to_csv.py’로 입력한 다음 저장한 다음 해당 파일을 실행합니다.

 import os
 import glob
 import pandas as pd
 import xml.etree.ElementTree as ET
 def xml_to_csv(path):
     xml_list = []
     for xml_file in glob.glob(path + '/*.xml'):
         tree = ET.parse(xml_file)
         root = tree.getroot()
         for member in root.findall('object'):
             value = (root.find('filename').text,
                      int(root.find('size')[0].text),
                      int(root.find('size')[1].text),
                      member[0].text,
                      int(member[4][0].text),
                      int(member[4][1].text),
                      int(member[4][2].text),
                      int(member[4][3].text)
                      )
             xml_list.append(value)
     column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
     xml_df = pd.DataFrame(xml_list, columns=column_name)
     return xml_df
 def main():
     for directory in ['train','test']:
         image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
         xml_df = xml_to_csv(image_path)
         xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
         print('Successfully converted xml to csv.')
 main()

‘object-detection’폴더에서 ‘data’라는 새로운 폴더를 생성 후 ‘object-detection’ 폴더 내에서 터미널을 열어 아래의 명령을 차례대로 실행

sudo pip3 install pandas
python3 xml_to_csv.py

‘object-detection’폴더에서 아래 그림과 같이 오른쪽 클릭을 한 다음 새로운 문서의 이름을 ‘generate_tfrecord.py’로 입력한 다음 저장한 다음 해당 파일을 실행.



"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
 
  # Create test data:
  python3 generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
 
import os
import io
import pandas as pd
import tensorflow as tf
 
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
 
flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS
 
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'macncheese':
        return 1
    else:
        None
 
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
 
def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
 
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
 
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
 
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
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
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), 'images')
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
 
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
 
 
if __name__ == '__main__':
    tf.app.run()

>## -> 해당 코딩이 오류가 발생할 시 main을 다음과 같이 변경한다.
def main(_):    
    print(os.getcwd())
    writer = tf.python_io.TFRecordWriter('data/train.record')
    # path = os.path.join(FLAGS.image_dir)
    path = 'images/train'
    # examples = pd.read_csv(FLAGS.csv_input)
    examples = pd.read_csv('data/train_labels.csv')
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

>## main문을 다음과 같이 변경했을 경우 test
   
‘Desktop/models’에서 터미널을 열어
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..
cd object-detection
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record

>### step5 : 기기학습

http://www.birc.co.kr/download/2803/  파일을 다운로드한 다음 ‘object-detection’ 폴더에 압축 해제

‘object-dection’ 폴더 내에 ‘training’ 폴더를 새롭게 생성하고 새롭게 생성된 ‘training’ 폴더 내에서 ‘ssd_mobilenet_v1_pets.config’ 라는 제목의 새로운 문서를 생성한다. 그리고 ‘ssd_mobilenet_v1_pets.config’를 실행하여 아래의 코드를 복사해서 붙여넣은 후 저장.
# SSD with Mobilenet v1, configured for the mac-n-cheese dataset.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "${YOUR_GCS_BUCKET}" to find the fields that
# should be configured.

model {
  ssd {
    num_classes: 1
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    anchor_generator {
      ssd_anchor_generator {
        num_layers: 6
        min_scale: 0.2
        max_scale: 0.95
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.3333
      }
    }
    image_resizer {
      fixed_shape_resizer {
        height: 300
        width: 300
      }
    }
    box_predictor {
      convolutional_box_predictor {
        min_depth: 0
        max_depth: 0
        num_layers_before_predictor: 0
        use_dropout: false
        dropout_keep_probability: 0.8
        kernel_size: 1
        box_code_size: 4
        apply_sigmoid_to_scores: false
        conv_hyperparams {
          activation: RELU_6,
          regularizer {
            l2_regularizer {
              weight: 0.00004
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.03
              mean: 0.0
            }
          }
          batch_norm {
            train: true,
            scale: true,
            center: true,
            decay: 0.9997,
            epsilon: 0.001,
          }
        }
      }
    }
    feature_extractor {
      type: 'ssd_mobilenet_v1'
      min_depth: 16
      depth_multiplier: 1.0
      conv_hyperparams {
        activation: RELU_6,
        regularizer {
          l2_regularizer {
            weight: 0.00004
          }
        }
        initializer {
          truncated_normal_initializer {
            stddev: 0.03
            mean: 0.0
          }
        }
        batch_norm {
          train: true,
          scale: true,
          center: true,
          decay: 0.9997,
          epsilon: 0.001,
        }
      }
    }
    loss {
      classification_loss {
        weighted_sigmoid {
          anchorwise_output: true
        }
      }
      localization_loss {
        weighted_smooth_l1 {
          anchorwise_output: true
        }
      }
      hard_example_miner {
        num_hard_examples: 3000
        iou_threshold: 0.99
        loss_type: CLASSIFICATION
        max_negatives_per_positive: 3
        min_negatives_per_image: 0
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    normalize_loss_by_num_matches: true
    post_processing {
      batch_non_max_suppression {
        score_threshold: 1e-8
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
  }
}

train_config: {
  batch_size: 10
  optimizer {
    rms_prop_optimizer: {
      learning_rate: {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.004
          decay_steps: 800720
          decay_factor: 0.95
        }
      }
      momentum_optimizer_value: 0.9
      decay: 0.9
      epsilon: 1.0
    }
  }
  fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
  from_detection_checkpoint: true
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    ssd_random_crop {
    }
  }
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "data/train.record"
  }
  label_map_path: "data/object-detection.pbtxt"
}

eval_config: {
  num_examples: 40
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "data/test.record"
  }
  label_map_path: "training/object-detection.pbtxt"
  shuffle: false
  num_readers: 1
}
    
‘training’ 폴더에서 ‘object-detection.pbtxt’ 라는 이름의 새로운 문서를 생성, ‘object-detection.pbtxt’를 실행한 후 아래의 코드를 복사하여 붙여넣은 다음 저장

item{
  id: 1
  name: 'macncheese'
}


‘object-detection.pbtxt’를 복사하여 ‘object-detection/data’에 붙여넣은 후 object-detection 내의 ‘data’, ‘images’, ‘ssd_mobilenet_v1_coco_11_06_2017’, ‘training’ 폴더를 ‘models/obejct_detection’ 폴더로 복사하여 붙여넣는다.

‘Desktop/models’에서 터미널을 실행한 다음 

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd object_detection 
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

>### step6 : 결과 확인
‘Desktop/models’에서 터미널을 실행한 다음
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd object_detection
python3 export_inference_graph.py \
 --input_type image_tensor \
 --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
 --trained_checkpoint_prefix training/model.ckpt-9540 \
 --output_directory mac_n_cheese_graph
 
위 코드중 ckpt 부분을 파일 중 가장 높은 수의 파일을 기준으로 작성된 코드를 기준으로 숫자를 변경한다.
‘Desktop/models/object_detection’에서 터미널 실행 후 jupyter notebook으로 실행한다. ‘object_detection_tutoral.ipynb’를 클릭하고 
Variables’ 코드를 
# What model to download.
MODEL_NAME = 'mac_n_cheese_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
NUM_CLASSES = 1

‘Download Model’ 코드는 삭제하고 ‘Desktop/object-detection/images’ 폴더로 이동하여 테스트 하고자 하는 몇 개의 이미지를 복사한 다음 ‘Desktop/models/object_detection/test_images’에 붙여넣습니다.붙여넣은 이미지 파일의 이름을 ‘image1.jpg’, ‘image2.jpg’ 와 같이 image + 숫자 + .jpg 로 변경합니다. Detection에 관한 코드를 이미지 개수에 맞게 변경합니다. 제 경우에는 image3.jpg ~ image5.jpg 가 test image로 활용되므로 for i in range(1,3)을 for i in range(3, 6)으로 변경했습니다.

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 6) ]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
‘Cell – Run All’
