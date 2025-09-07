import tensorflow as tf
from tensorflow.keras  import layers

class ARSLIFLayer(tf.keras.layers.Layer): 
    def __init__(self, units, f=0.65, f_adapt=0.85, f_active=0.99, f_rest=0.01, 
                 tau_adapt=1000, tau_mem=10, dt=1, alpha=0.6, beta=0.4, 
                 surrogate_slope=1, **kwargs):
        super(ARSLIFLayer, self).__init__(**kwargs)
        self.units  = units
        self.f = f
        self.f_adapt = f_adapt
        self.f_active = f_active
        self.f_rest = f_rest
        self.tau_adapt  = tau_adapt
        self.tau_mem  = tau_mem
        self.dt  = dt
        self.alpha  = alpha
        self.beta  = beta
        self.surrogate_slope  = surrogate_slope

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight("kernel",  shape=(input_dim, self.units)) 
        self.b = self.add_weight("bias",  shape=(self.units,)) 
        self.R = self.add_weight("R",  shape=(self.units,),  
                                initializer=tf.initializers.RandomUniform(minval=0.1,  maxval=1.0))

    def spike_function(self, v):
        return tf.nn.sigmoid(self.surrogate_slope  * (v - 1.0))

    @tf.custom_gradient 
    def spike_gradient(self, v):
        z = self.spike_function(v) 
        def grad(dy):
            dz_dv = z * (1 - z) * self.surrogate_slope 
            return dy * dz_dv
        return tf.cast(v  >= 1.0, tf.float32),  grad

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0] 
        
        V_mem = tf.zeros((batch_size,  self.units)) 
        
        I = tf.matmul(inputs,  self.W) + self.b
        
        V_th = tf.abs(I)  / self.f_adapt
        
        dV = (self.dt  / self.tau_mem)  * (-V_mem + self.R * I)
        
        V_mem_new = V_mem + dV
        
        v_scaled = V_mem_new / (V_th + 1e-7)
        
        spikes = self.spike_gradient(v_scaled) 
        
        V_mem = tf.where(spikes  > 0, tf.zeros_like(V_mem_new),  V_mem_new)
        
        r = tf.reduce_mean(spikes,  axis=1, keepdims=True)
        
        f_adapt = tf.where(tf.abs(self.R  * I) >= V_th, self.f_active, self.f_rest)
        
        r_ei = r - self.f
        r_ea = r - tf.reduce_mean(f_adapt,  axis=1, keepdims=True)
        
        r_e = self.alpha  * r_ei + self.beta  * r_ea
        
        dV_th = (self.dt  / self.tau_adapt)  * r_e
        
        V_th += dV_th
        
        return spikes

    def get_config(self):
        config = super().get_config()
        config.update({ 
            "units": self.units, 
            "f": self.f,
            "f_adapt": self.f_adapt,
            "f_active": self.f_active,
            "f_rest": self.f_rest,
            "tau_adapt": self.tau_adapt, 
            "tau_mem": self.tau_mem, 
            "dt": self.dt, 
            "alpha": self.alpha, 
            "beta": self.beta, 
            "surrogate_slope": self.surrogate_slope 
        })
        return config

class ARSLIFActivation(layers.Layer):
    def __init__(self, units, **kwargs):
        super(ARSLIFActivation, self).__init__(**kwargs)
        self.units  = units
        self.arslif_layer  = ARSLIFLayer(units)

    def build(self, input_shape):
        self.arslif_layer.build(input_shape) 
        super(ARSLIFActivation, self).build(input_shape)

    def call(self, inputs):
        return self.arslif_layer(inputs) 

    def get_config(self):
        config = super().get_config()
        config.update({ 
            "units": self.units 
        })
        return config

class ARSLIFLSTMCell(tf.keras.layers.LSTMCell): 
    def __init__(self, units, f=0.65, f_adapt=0.85, f_active=0.99, f_rest=0.01, 
                 tau_adapt=1000, tau_mem=10, dt=1, alpha=0.6, beta=0.4, 
                 surrogate_slope=1, **kwargs):
        super(ARSLIFLSTMCell, self).__init__(units, **kwargs)
        self.units  = units
        self.f = f
        self.f_adapt = f_adapt
        self.f_active = f_active
        self.f_rest = f_rest
        self.tau_adapt  = tau_adapt
        self.tau_mem  = tau_mem
        self.dt  = dt
        self.alpha  = alpha
        self.beta  = beta
        self.surrogate_slope  = surrogate_slope
        self.state_size  = [tf.TensorShape([units]), tf.TensorShape([units]), 
                           tf.TensorShape([units * 4]), tf.TensorShape([units * 4])]

    def build(self, input_shape):
        super(ARSLIFLSTMCell, self).build(input_shape)
        self.R = self.add_weight("R",  shape=(self.units  * 4,), initializer="ones")

    def spike_function(self, v):
        return tf.nn.sigmoid(self.surrogate_slope  * (v - 1.0))

    @tf.custom_gradient 
    def spike_gradient(self, v):
        z = self.spike_function(v) 
        def grad(dy):
            dz_dv = z * (1 - z) * self.surrogate_slope 
            return dy * dz_dv
        return z, grad

    def call(self, inputs, states, training=None):
        inputs = tf.cast(inputs,  tf.float32) 

        h_tm1, c_tm1, V_mem, V_th = [tf.cast(state, tf.float32)  for state in states]

        if 0 < self.dropout  < 1.:
            inputs = self.dropout_layer(inputs,  training=training)
        
        z = tf.keras.backend.dot(inputs,  self.kernel) 
        
        if 0 < self.recurrent_dropout  < 1.:
            h_tm1 = self.recurrent_dropout_layer(h_tm1,  training=training)
        
        z += tf.keras.backend.dot(h_tm1,  self.recurrent_kernel) 

        if self.use_bias: 
            z = tf.keras.backend.bias_add(z,  self.bias) 

        dV = (self.dt  / self.tau_mem)  * (-V_mem + self.R * z)
        
        V_mem_new = V_mem + dV
        
        v_scaled = V_mem_new / (V_th + 1e-7)
        
        spikes = self.spike_gradient(v_scaled) 

        V_mem_new = tf.where(spikes  > 0, tf.zeros_like(V_mem_new),  V_mem_new)

        r = tf.reduce_mean(spikes,  axis=1, keepdims=True)
        r = tf.clip_by_value(r,  0, 1)

        f_adapt = tf.where(tf.abs(self.R  * z) >= V_th, self.f_active, self.f_rest)
        
        r_ei = r - self.f
        r_ea = r - tf.reduce_mean(f_adapt,  axis=1, keepdims=True)
        
        r_e = self.alpha  * r_ei + self.beta  * r_ea
        
        dV_th = (self.dt  / self.tau_adapt)  * r_e
        
        V_th_new = V_th + dV_th

        z = spikes
        
        z0, z1, z2, z3 = tf.split(z,  num_or_size_splits=4, axis=1)

        i = self.recurrent_activation(z0) 
        f = self.recurrent_activation(z1) 
        c = f * c_tm1 + i * self.activation(z2) 
        o = self.recurrent_activation(z3) 

        h = o * self.activation(c) 
        
        return h, [h, c, V_mem_new, V_th_new]

    def get_config(self):
        config = super().get_config()
        config.update({ 
            "f": self.f,
            "f_adapt": self.f_adapt,
            "f_active": self.f_active,
            "f_rest": self.f_rest,
            "tau_adapt": self.tau_adapt, 
            "tau_mem": self.tau_mem, 
            "dt": self.dt, 
            "alpha": self.alpha, 
            "beta": self.beta, 
            "surrogate_slope": self.surrogate_slope 
        })
        return config

class ARSLIFBiLSTM(layers.Layer):
    def __init__(self, units, **kwargs):
        super(ARSLIFBiLSTM, self).__init__(**kwargs)
        self.units  = units
        
        # 创建带有ARSLIF激活的LSTM层
        self.forward_lstm  = layers.RNN(ARSLIFLSTMCell(units), return_sequences=True)
        self.backward_lstm  = layers.RNN(ARSLIFLSTMCell(units), return_sequences=True, go_backwards=True)

    def call(self, inputs):
        forward_outputs = self.forward_lstm(inputs) 
        
        backward_outputs = self.backward_lstm(inputs) 

        x = tf.concat([forward_outputs,  backward_outputs], axis=-1)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({ 
            "units": self.units 
        })
        return config