# # multivariate version

# import torch
# import torch.nn as nn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class Encoder(nn.Module):
#     def __init__(self, d_input:int, d_model:int, n_layers:int, dr:float):
#         # d_embedding:int, n_embedding:list, 
#         super(Encoder, self).__init__()
#         # self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
#         # + len(n_embedding) * d_embedding
#         self.lstm = nn.LSTM(d_input , d_model, n_layers, dropout=dr, batch_first=True)
        
#     def forward(self, conti): # , cate
#         # tmp_feature_list = []
        
#         # for i, l in enumerate(self.embedding_layers):
#         #     tmp_feature = l(cate[:, :, i:i+1])
#         #     tmp_feature_list.append(tmp_feature)
            
#         # emb_output = torch.cat(tmp_feature_list, axis=-2)
#         # emb_output = emb_output.view(conti.size(0), conti.size(1), -1)
        
#         # x = torch.cat([conti, emb_output], axis=-1)
        
#         _, (hidden, cell) = self.lstm(conti) # x

#         return hidden, cell # (num_layers, batch_size, d_model)
    
# class GlobalDecoder(nn.Module):
#     def __init__(self, d_hidden:int, d_model:int, tau:int, num_targets:int, dr:float):
#         # d_embedding:int, n_embedding:list, 
#         super(GlobalDecoder, self).__init__()
#         self.d_hidden = d_hidden
#         # self.d_embedding = d_embedding
#         # self.n_embedding = n_embedding
#         self.d_model = d_model
#         self.tau = tau
#         self.num_targets = num_targets
#         self.dr = dr
#         # self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
#         self.linear_layers = nn.ModuleList([nn.Linear(d_hidden + tau * d_model, (tau+1) * d_model) for _ in range(num_targets)])
#         # * d_embedding * len(n_embedding)
#         self.dropout = nn.Dropout(dr)
        
#     def forward(self, future, hidden):
#         # tmp_feature_list = []
        
#         # for i, l in enumerate(self.embedding_layers):
#         #     tmp_feature = l(future[:, :, i:i+1])
#         #     tmp_feature_list.append(tmp_feature)
            
#         # emb_output_ = torch.cat(tmp_feature_list, axis=-2)
#         # emb_output = emb_output_.view(future.size(0), -1)
        
#         # num_layers, batch_size, d_hidden = hidden.size()
        
#         # assert d_hidden == self.d_model 
#         batch_size = future.size(0)

#         # Assuming future is a tensor indicating the treated unit
#         cate_tensor = future.view(batch_size, -1, 1)

#         num_layers, _, _ = hidden.size()

#         assert self.d_hidden == self.d_model
        
#         x = torch.cat([hidden[num_layers-1], cate_tensor], axis=-1) #emb_output
        
#         tmp_global_context = []
#         for l in self.linear_layers:
#             tmp_gc = self.dropout(l(x))
#             tmp_global_context.append(tmp_gc.unsqueeze(1))
        
#         global_context = torch.cat(tmp_global_context, dim=1) # , axis=1
        
#         return cate_tensor, global_context # (batch_size, tau, d_embedding * len(n_embedding)), (batch_size, num_targets, (tau+1) * d_model), (tau+1): c_{a} , c_{t+1:t+tau}
#         # emb_output_.view(batch_size, self.tau, -1)
# class LocalDecoder(nn.Module):
#     def __init__(self, d_hidden:int, d_model:int, tau:int, num_targets:int, num_quantiles:int, dr:float):
#         # , d_embedding:int, n_embedding: list
#         super(LocalDecoder, self).__init__()
#         self.d_hidden = d_hidden
#         # self.d_embedding = d_embedding
#         # self.n_embedding = n_embedding
#         self.d_model = d_model
#         self.tau = tau
#         self.num_targets = num_targets
#         self.dr = dr
#         self.linear_layers = nn.Sequential(
#             nn.Linear(2 * d_model, d_model * 2), # + d_embedding * len(n_embedding)
#             nn.Dropout(dr),
#             nn.Linear(d_model * 2, d_model),
#             nn.Dropout(dr),
#             nn.Linear(d_model, num_quantiles)            
#             )
                
#     def forward(self, embedded_future, global_output):
#         batch_size = global_output.size(0)
        
#         c_a = global_output[..., :self.d_model].unsqueeze(-2).repeat(1, 1, self.tau, 1) # (batch_size, num_targets, tau, d_model)
#         c_t = global_output[..., self.d_model:].view(batch_size, self.num_targets, self.tau, -1) # (batch_size, num_targets, tau, d_model)
#         x_ = torch.cat([c_a,c_t.view(batch_size, self.num_targets, self.tau, -1)], axis=-1) # (batch_size, num_targets, tau, 2*d_model)
#         x = torch.cat([x_, embedded_future.unsqueeze(1).repeat(1, self.num_targets, 1, 1)], axis=-1) # (batch_size, num_targets, tau, 2*d_model + d_embedding * len(n_embedding))
        
#         output = self.linear_layers(x)
        
#         return output # (batch_size, num_targets, tau, num_quantiles)

# class MQRNN(nn.Module):
#     def __init__(self, d_input:int, d_model:int, tau:int, num_targets:int, num_quantiles: int, n_layers:int, dr:float):
#         # d_embedding:int, n_embedding:list, 
#         super(MQRNN, self).__init__()
#         self.encoder = Encoder(
#                                d_input=d_input,
#                             #    d_embedding=d_embedding,
#                             #    n_embedding=n_embedding,
#                                d_model=d_model,
#                                n_layers=n_layers,
#                                dr=dr
#                                )
#         self.global_decoder = GlobalDecoder(
#                                             d_hidden=d_model,
#                                             # d_embedding=d_embedding,
#                                             # n_embedding=n_embedding,
#                                             d_model=d_model,
#                                             tau=tau,
#                                             num_targets=num_targets,
#                                             dr=dr
#                                             )
#         self.local_decoder = LocalDecoder(
#                                           d_hidden=d_model,
#                                         #   d_embedding=d_embedding,
#                                         #   n_embedding=n_embedding,
#                                           d_model=d_model,
#                                           tau=tau,
#                                           num_targets=num_targets,
#                                           num_quantiles=num_quantiles,
#                                           dr=dr
#                                           )
        
#     def forward(self, conti, cate, future):
#         hidden, _ = self.encoder(conti, cate)
#         embedded_future, global_output = self.global_decoder(future, hidden)
#         output = self.local_decoder(embedded_future, global_output)
        
#         return output

import tensorflow as tf
from tensorflow.keras import layers

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_layers, dr):
        super(Encoder, self).__init__()
        # Define your model
        self.lstm = tf.keras.Sequential()
        for _ in range(n_layers):
            self.lstm.add(tf.keras.layers.LSTM(units=d_model, return_state=True, dropout=dr))

    def call(self, conti, cate):
        # Use cate in the encoder
        x = tf.concat([conti, cate], axis=1)

        _, hidden, cell = self.lstm(x)
        # _, hidden, cell = self.lstm(inputs)
        return hidden, cell

class GlobalDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model, tau, num_targets, dr):
        super(GlobalDecoder, self).__init__()
        self.d_model = d_model
        self.tau = tau
        self.num_targets = num_targets
        self.dr = dr
        self.linear_layers = [tf.keras.layers.Dense((tau + 1) * d_model) for _ in range(num_targets)]
        self.dropout = tf.keras.layers.Dropout(dr)

    def call(self, future, hidden):
        batch_size = tf.shape(future)[0]

        # Assuming future is a tensor indicating the treated unit
        cate_tensor = tf.expand_dims(future, axis=-1)

        num_layers, _, _ = tf.shape(hidden)

        assert self.d_model == num_layers

        x = tf.concat([hidden[num_layers - 1], cate_tensor], axis=-1)

        tmp_global_context = []
        for l in self.linear_layers:
            tmp_gc = self.dropout(l(x))
            tmp_global_context.append(tf.expand_dims(tmp_gc, axis=1))

        global_context = tf.concat(tmp_global_context, axis=1)

        return cate_tensor, global_context

class LocalDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model, tau, num_targets, num_quantiles, dr):
        super(LocalDecoder, self).__init__()
        self.d_model = d_model
        self.tau = tau
        self.num_targets = num_targets
        self.dr = dr
        self.linear_layers = [
            tf.keras.layers.Dense(d_model * 2),
            tf.keras.layers.Dropout(dr),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dr),
            tf.keras.layers.Dense(num_quantiles)
        ]

    def call(self, embedded_future, global_output):
        batch_size = tf.shape(global_output)[0]

        c_a = tf.tile(tf.expand_dims(global_output[..., :self.d_model], axis=-2), [1, 1, self.tau, 1])
        c_t = tf.reshape(global_output[..., self.d_model:], [batch_size, self.num_targets, self.tau, -1])
        x_ = tf.concat([c_a, c_t], axis=-1)
        x = tf.concat([x_, tf.expand_dims(embedded_future, axis=1)], axis=-1)

        output = self.linear_layers(x)

        return output

class MQRNN(tf.keras.Model):
    def __init__(self, d_model, tau, num_targets, num_quantiles, n_layers, dr):
        super(MQRNN, self).__init__()
        self.encoder = Encoder(d_model, n_layers, dr)
        self.global_decoder = GlobalDecoder(d_model, tau, num_targets, dr)
        self.local_decoder = LocalDecoder(d_model, tau, num_targets, num_quantiles, dr)

    def call(self, conti, cate, future):
        hidden, _ = self.encoder(conti, cate)
        embedded_future, global_output = self.global_decoder(future, hidden)
        output = self.local_decoder(embedded_future, global_output)

        return output
