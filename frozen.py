import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

model = tf.saved_model.load('./saved_model/')
frozen_func = convert_variables_to_constants_v2(model.signatures['serving_default'], lower_control_flow=False)
graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
tf.io.write_graph(graph_def, './', 'frozen_model.pb', as_text=False)