"""
使用 tf2onnx.convert.from_saved_model 导出 ONNX
用法：python convert_savedmodel_to_onnx.py --saved_model artifacts/saved_models/dcn --output artifacts/onnx_models/dcn.onnx
"""
import argparse
import tf2onnx
import tensorflow as tf
print(tf2onnx.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--saved_model', required=True)
    # parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.saved_model = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/artifacts/saved_models/dcn/saved_model.pb'
    args.output = '/Users/honglei.yu/Documents/bmj/JHK/adjoe/artifacts/onnx_models/dcn.onnx'
    print('Converting', args.saved_model, '->', args.output)

    # # graph_def = tf.compat.v1.GraphDef()
    # graph_def = tf.compat.v1.GraphDef()
    # with tf.io.gfile.GFile(args.saved_model, "rb") as f:
    #     graph_def.ParseFromString(f.read())
    #
    # with tf.compat.v1.Session() as sess:
    #     tf.import_graph_def(graph_def, name='')
    #     g = sess.graph
    #     onnx_model = tf2onnx.convert.from_session(sess, input_names = ["input:0"], output_names = ["output:0"])
    #     with open(args.output, "wb") as f:
    #         f.write(onnx_model.SerializeToString())

    # model_proto, external_tensor_storage = tf2onnx.convert(args.saved_model, output_path=args.output)

    # 命令行，修改几个文件去掉np的几个函数object、bool
    # python -m tf2onnx.convert --saved-model /Users/honglei.yu/Documents/bmj/JHK/adjoe/artifacts/saved_models/dcn --output /Users/honglei.yu/Documents/bmj/JHK/adjoe/artifacts/onnx_models/dcn.onnx --opset 13

    print('Done')