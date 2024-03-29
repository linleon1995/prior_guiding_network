import tensorflow as  tf
from tensorflow.contrib import slim
from core import preprocess_utils


class spatial_attention(object):
	"""
	The class used for create spatial attention layer.
	By modifying the function, the user can define attention
 	in different way, e.g., channel attention.
 	"""
	def __init__(self, embed_node):
		self.embed_node = embed_node

	def __call__(self, f, g, h, scope):
		with tf.variable_scope(scope, 'self_attention'):
			self.n, self.h, self.w, self.c = preprocess_utils.resolve_shape(f, rank=4)
			f = self.embedding(f, "f") # [bs, h, w, emb_c]
			g = self.embedding(g, "g")
			h = self.embedding(h, "h")

			# N = h * w
			o = self.get_attention(self.flatten(f), self.flatten(g), self.flatten(h))

			o = tf.reshape(o, shape=tf.shape(f)) # [bs, h, w, emb_c]
			y = f + self.embedding(o, "y")
			return y

	def embedding(self, x, scope):
		return slim.conv2d(x, self.embed_node, kernel_size=[1, 1], stride=1, activation_fn=None, scope=scope)

	def get_attention(self, f, g, h):
		s = tf.matmul(f, g, transpose_b=True)
		beta = tf.nn.softmax(s)  # [bs, N, N]
		o = tf.matmul(beta, h) # [bs, N, emb_c]
		return o

	def flatten(self, x):
		return tf.reshape(x, [-1, self.h*self.w, self.embed_node])


class channel_attention(spatial_attention):
	def flatten(self, x):
		x = tf.reshape(x, [self.n, self.h*self.w, self.embed_node])
		return tf.transpose(x, [0, 2, 1])

	def get_attention(self, f, g, h):
		s = tf.matmul(f, g, transpose_b=True)
		beta = tf.nn.softmax(s)  # [bs, emb_c, emb_c]
		o = tf.matmul(beta, h) # [bs, emb_c, N]
		o = tf.transpose(o, [0, 2, 1])
		return o