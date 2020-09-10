
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet101, ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2



# model = ResNet50(weights=None)
model = ResNet101()
# model = MobileNet()
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec([None, 224, 224, 3], model.input[0].dtype))

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)  # [<tf.Tensor 'x:0' shape=(None, 224, 224, 3) dtype=float32>]
print("Frozen model outputs: ")
print(frozen_func.outputs)  # [<tf.Tensor 'Identity:0' shape=(None, 1000) dtype=float32>]


tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="resnet101.pb",
                  as_text=False)


