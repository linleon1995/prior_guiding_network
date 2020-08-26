import tensorflow as  tf
from tensorflow.contrib import slim
from core import preprocess_utils
# a = self_attention()
# a.attention(x, 32, "x1")
class self_attention(object):
    def __init__(self):
        pass
        # [bs, h, w, c]
        # self.n, self.h, self.w, self.c = resolve_shape(x, rank=4)
        # self.n, self.h, self.w, self.c = x.get_shape.as_list()
        # self.scope = scope
        # self.attention(x, emb_c)
    
    def embedding(self, x, channel, scope):
        return slim.conv2d(x, channel, kernel_size=[1, 1], stride=1, activation_fn=None, scope=scope)
    
    def get_attention(self, f, g, h):
        s = tf.matmul(g, f, transpose_b=True)
        beta = tf.nn.softmax(s)  # [bs, N, N]
        o = tf.matmul(beta, h) # [bs, N, emb_c]
        return o
    
    def flatten(self, x):
        print(self.n, self.h, self.w, self.c)
        return tf.reshape(x, [-1, self.h*self.w, self.c])
        
    def attention(self, x, emb_c, scope):
        with tf.variable_scope(scope, 'self_attention'):
            self.n, self.h, self.w, self.c = preprocess_utils.resolve_shape(x, rank=4)
            
            f = self.embedding(x, emb_c, "f") # [bs, h, w, emb_c]
            g = self.embedding(x, emb_c, "g")
            h = self.embedding(x, emb_c, "h")
            
            # N = h * w
            o = self.get_attention(self.flatten(f), self.flatten(g), self.flatten(h)) 
            
            # shape = x.shape.as_list()[:3].append(emb_c)
            # o = tf.reshape(o, shape=shape) # [bs, h, w, emb_c]
            o = tf.reshape(o, shape=f.shape) # [bs, h, w, emb_c]
            y = self.embedding(o, emb_c, "y")
            return y
    

# class channel_attention(self_attention):
#     def __init__(self, x1, x2, emb_c, scope):
#         super().__init__(x1, channel, scope)
        
#     def get_attention(self, f, g, h):
#         s = tf.matmul(g, f, transpose_b=True)
#         beta = tf.nn.softmax(s)  # [bs, N, N]
#         o = tf.matmul(beta, h) # [bs, N, emb_c]
#         return o
    
class context_attention(self_attention):
    def attention(self, feat, context, emb_c, scope):
        with tf.variable_scope(scope, 'context_attention'):
            self.n, self.h, self.w, self.c = feat.get_shape().as_list()
            f = self.embedding(feat, emb_c, "f") # [bs, h, w, emb_c]
            g = self.embedding(context, emb_c, "g")
            h = self.embedding(feat, emb_c, "h")
            
            # N = h * w
            o = self.get_attention(self.flatten(f), self.flatten(g), self.flatten(h)) 
            
            # shape = x.shape.as_list()[:3].append(emb_c)
            # o = tf.reshape(o, shape=shape) # [bs, h, w, emb_c]
            o = tf.reshape(o, shape=tf.shape(f)) # [bs, h, w, emb_c]
            y = self.embedding(o, emb_c, "y")
            return y
            
# class object_attention(self_attention):
#     pass