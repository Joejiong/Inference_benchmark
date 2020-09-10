import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

tf.compat.v1.keras.backend.set_session(
    sess
)


img = tf.compat.v1.placeholder(tf.float32, shape=(None, 2), name='input_plhdr')


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', name='Intermediate'),
    tf.keras.layers.Dense(2, activation='softmax', name='Output'),
])

M = model(img)
print('input', img.name)
print('output', M.name)
sess.run(tf.compat.v1.global_variables_initializer())
print('result', sess.run(M, {img: np.array([[42, 43.]], dtype=np.float32)}))

saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
saver.save(sess, './exported/my_model')
tf.compat.v1.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)