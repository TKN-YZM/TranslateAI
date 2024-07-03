from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, CuDNNGRU
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
from tokenizer import TokenizerWrap



mark_start = 'ssss '
mark_end = ' eeee'

data_src = []
data_dest = []


for line in open('tur.txt', encoding='UTF-8'):
    en_text, tr_text = line.rstrip().split('\t')
    
    tr_text = mark_start + tr_text + mark_end
    
    data_src.append(en_text)
    data_dest.append(tr_text)
    

    
    
tokenizer_src = TokenizerWrap(texts=data_src,
                              padding='pre',
                              reverse=True,
                              num_words=None)

tokenizer_dest = TokenizerWrap(texts=data_dest,
                              padding='post',
                              reverse=False,
                              num_words=None)


tokens_src = tokenizer_src.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded

decoder_input_data = tokens_dest[:, :-1]
decoder_output_data = tokens_dest[:, 1:]


word2vec = {}
with open('glove.6B.100d.txt', encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

embedding_matrix = np.random.uniform(-1, 1, (num_encoder_words, embedding_size))
for word, i in tokenizer_src.word_index.items():
    if i < num_encoder_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
            
encoder_input = Input(shape=(None,), name='encoder_input')
            
            
encoder_embedding = Embedding(input_dim=num_encoder_words,
                  output_dim=embedding_size,
                  weights=[embedding_matrix],
                  trainable=True,
                  name='encoder_embedding')

state_size = 256

encoder_gru1 = GRU(state_size, name='encoder_gru1', return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2', return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3', return_sequences=False)



def connect_encoder():
    net = encoder_input
    
    net = encoder_embedding(net)
    
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)
    
    encoder_output = net
    
    return encoder_output
            
            
    
encoder_output = connect_encoder()
decoder_initial_state = Input(shape=(state_size,), name='decoder_initial_state')
decoder_input = Input(shape=(None,), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_decoder_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1', return_sequences=True)

decoder_dense = Dense(num_decoder_words,
                      activation='linear',
                      name='decoder_output')
decoder_gru2 = GRU(state_size, name='decoder_gru2', return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3', return_sequences=True)
            
def connect_decoder(initial_state):
    net = decoder_input
    
    net = decoder_embedding(net)
    
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    
    decoder_output = decoder_dense(net)
    
    return decoder_output            

decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])


model_encoder = Model(inputs=[encoder_input], outputs=[encoder_output])


decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state], outputs=[decoder_output])


def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

optimizer = RMSprop(lr=1e-3)


decoder_target = tf.placeholder(dtype='int32', shape=(None,None))



model_train.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])


path_checkpoint = 'checkpoint.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, save_weights_only=True)

try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print('Checkpoint yüklenirken hata oluştu. Eğitime sıfırdan başlanıyor.')
    print(error)
    
x_data = {'encoder_input': encoder_input_data, 'decoder_input': decoder_input_data}

y_data = {'decoder_output': decoder_output_data}

model_train.fit(x=x_data,
                y=y_data,
                batch_size=256,
                epochs=0,
                callbacks=[checkpoint])


def translate(input_text, true_output_text=None):
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding='pre')
    
    initial_state = model_encoder.predict(input_tokens)
    
    max_tokens = tokenizer_dest.max_tokens
    
    decoder_input_data = np.zeros(shape=(1, max_tokens), dtype=np.int)
    
    token_int = token_start
    output_text = ''
    count_tokens = 0
    
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = {'decoder_initial_state': initial_state, 'decoder_input': decoder_input_data}
        
        decoder_output = model_decoder.predict(x_data)
        
        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        
        sampled_word = tokenizer_dest.token_to_word(token_int)
        output_text += ' ' + sampled_word
        count_tokens += 1
        
    print('Input text:')
    print(input_text)
    print()
    
    print('Translated text:')
    print(output_text)
    print()
    
    if true_output_text is not None:
        print('True output text:')
        print(true_output_text)
        print()
        


translate(input_text='Which road leads to the airport?')

'''
Input text:
Which road leads to the airport?

Translated text:
 hangi yol havaalanına gider eeee
 '''




