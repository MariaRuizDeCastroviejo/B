#!/usr/bin/env python
# coding: utf-8

# # Traducción de texto mediante algoritmos de Machine learning

# ### Importación de las librerías

# In[1]:


import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, RepeatVector, GRU, SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 200)


# ### Lectura de datos

# Los datos son un archivo de texto de pares de oraciones en inglés-español.

# In[2]:


# Leemos el archivo
def read_text(filename):
    # open the file
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    file.close()
    return text


# Ahora definamos una función para dividir el texto en pares inglés-español separados por '\ n'. Luego dividimos estos pares en oraciones de inglés y español.

# In[3]:


# Dividimos en frases
def to_lines(text):
    sents = text.strip().split('\n')
    sents = [i.split('\t') for i in sents]
    return sents


# Para descargar los datos pulse [aquí.](http://www.manythings.org/anki/deu-eng.zip)__ y extrae "spa.txt" en tu directorio de trabajo.

# In[4]:


data = read_text("C:/Users/María/MASTER/Datos No Estruct/Texto/Practica/data/spa.txt")
spa_eng = to_lines(data)
spa_eng = array(spa_eng)


# Los datos contienen más de 150 mil pares de oraciones. Sin embargo, usaremos los primeros 50 mil pares de oraciones para reducir el tiempo de entrenamiento del modelo.

# In[5]:


spa_eng = spa_eng[:50000,:]


# In[6]:


spa_eng.shape


# ### Limpieza de texto

# Echemos un vistazo a nuestros datos:

# In[7]:


spa_eng


# Eliminaremos los signos de puntuación y luego convertiremos el texto a minúsculas.

# In[8]:


# Eliminamos los signos de puntuación 
spa_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in spa_eng[:,0]]
spa_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in spa_eng[:,1]]


# In[9]:


spa_eng


# In[10]:


# Convertimos el texto a minúsculas
for i in range(len(spa_eng)):
    spa_eng[i,0] = spa_eng[i,0].lower()
    
    spa_eng[i,1] = spa_eng[i,1].lower()


# In[11]:


spa_eng


# ### Preprocesamiento de texto

# - #### Conversión de texto a secuencia

# Para alimentar nuestros datos en un modelo *Seq2Seq*, tendremos que convertir las oraciones de entrada y de salida en secuencias enteras de longitud fija. Antes de eso, visualizamos la longitud de las oraciones en dos listas separadas para inglés y para español, respectivamente.

# In[12]:


# Listas vacias
eng_l = []
spa_l = []

# Rellenar las listas con longitudes de oraciones
for i in spa_eng[:,0]:
    eng_l.append(len(i.split()))

for i in spa_eng[:,1]:
    spa_l.append(len(i.split()))


# In[13]:


length_df = pd.DataFrame({'Inglés':eng_l, 'Español':spa_l})
print(length_df)
print(max(length_df.Inglés))
print(max(length_df.Español))


# In[14]:


length_df.hist(bins = 30)
plt.show()


# La longitud máxima de las frases en Español es de 10 y la de las frases en Inglés es de 7.

# - #### Tokenización

# A continuación, necesitamos tokenizar los datos, es decir, convertir el texto en valores numéricos. Esto permite que la red neuronal realice operaciones sobre los datos de entrada. 

# Vectoricemos nuestros datos de texto usando la clase *Tokenizer ()* de Keras. Convertirá nuestras oraciones en secuencias de números enteros. Luego rellenaremos esas secuencias con ceros para hacer que todas las secuencias tengan la misma longitud.

# In[15]:


# Función para construir un tokenizador
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[16]:


# Preparar tokenizador en inglés
eng_tokenizer = tokenization(spa_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 8
print('Tamaño del vocabulario en inglés: %d' % eng_vocab_size)


# In[17]:


# Preparar tokenizador en español
spa_tokenizer = tokenization(spa_eng[:, 1])
spa_vocab_size = len(spa_tokenizer.word_index) + 1

spa_length = 8
print('Tamaño del vocabulario en español: %d' % spa_vocab_size)


# In[18]:


print(spa_tokenizer.word_counts)


# A continuación se muestra una función para preparar las secuencias. También realizará el relleno de secuencia hasta su longitud máxima.

# Cuando introducimos nuestras secuencias de ID de palabras en el modelo, cada secuencia debe tener la misma longitud. Para lograr esto, se agrega relleno a cualquier secuencia que sea más corta que la longitud máxima (es decir, más corta que la oración más larga).

# In[19]:


# Codificar y rellenar secuencias
def encode_sequences(tokenizer, length, lines):
    # Secuencias de codificación entera
    seq = tokenizer.texts_to_sequences(lines)
    # Rellenar las secuencias con 0
    seq = pad_sequences(seq, maxlen=length, padding='post')
    print(seq)
    print(len(seq))
    return seq


# eng_padding= encode_sequences(eng_tokenizer, 12, spa_eng[:, 0])

# ### Creación de la muestra de train y test

# Dividimos los datos para el entrenamiento y la evaluación del modelo.

# In[20]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(spa_eng, test_size=0.2, random_state = 12)


# Codificamos las oraciones de español como secuencias de entrada y las oraciones de inglés como secuencias de destino. Para el conjunto de train y test.

# In[21]:


# Preparación de los datos de entrenamiento
trainX = encode_sequences(spa_tokenizer, spa_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])


# In[22]:


print(trainX.shape)
print(trainY.shape)


# In[23]:


# Preparación de los datos de test
testX = encode_sequences(spa_tokenizer, spa_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])


# In[24]:


print(testX.shape)
print(testY.shape)


# ### Construcción del modelo

# Partes de la arquitectura de un RNN en un nivel alto:
# 
# - **Entradas.** Las secuencias de entrada se introducen en el modelo con una palabra para cada paso de tiempo. Cada palabra está codificada como un número entero único o un vector codificado en caliente que se asigna al vocabulario del conjunto de datos en inglés.
# 
# 
# - **Embeddings.** Las incrustaciones se utilizan para convertir cada palabra en un vector. El tamaño del vector depende de la complejidad del vocabulario.
# 
# 
# - **Capas recurrentes (codificador).** Aquí es donde el contexto de los vectores de palabras en los pasos de tiempo anteriores se aplica al vector de palabras actual.
# 
# 
# - **Capas densas (decodificador).** Estas son capas típicas completamente conectadas que se utilizan para decodificar la entrada codificada en la secuencia de traducción correcta.
# 
# 
# - **Salidas.** Las salidas se devuelven como una secuencia de números enteros o vectores codificados en caliente que luego se pueden asignar al vocabulario del conjunto de datos.

# Los **Embeddings** permiten capturar relaciones de palabras sintácticas y semánticas más precisas proyectando cada palabra en un espacio n-dimensional. Las palabras con significados similares ocupan regiones similares de este espacio; cuanto más cercanas están dos palabras, más similares son. Y a menudo los vectores entre palabras representan relaciones útiles, como género, tiempo verbal o incluso relaciones geopolíticas.
# 
# El entrenamiento con Embeddings en un gran conjunto de datos desde cero requiere una gran cantidad de datos y cálculos. Entonces, en lugar de hacerlo nosotros mismos, normalmente usamos un paquete de Embeddings previamente entrenado como word2vec . Cuando se usan de esta manera, las incrustaciones son una forma de aprendizaje por transferencia. Sin embargo, dado que nuestro conjunto de datos para este proyecto tiene un vocabulario pequeño y poca variación sintáctica, usaremos Keras para entrenar las incrustaciones nosotros mismos.

# Nuestro **modelo secuencia a secuencia** vincula dos redes recurrentes: un codificador y un descodificador. El codificador resume la entrada en una variable de contexto. Depués, se decodifica este contexto y se genera la secuencia de salida. Dado que tanto el codificador como el descodificador son recurrentes, tienen bucles que procesan cada parte de la secuencia en diferentes pasos de tiempo. 
# 
# Para cada paso de tiempo después de la primera palabra en la secuencia hay dos entradas: el estado oculto y una palabra de la secuencia. Para el codificador, es la siguiente palabra en la secuencia de entrada. Para el decodificador, es la palabra anterior de la secuencia de salida.
# 
# El aprendizaje secuencia a secuencia (Seq2Seq) se trata de entrenar modelos para convertir secuencias de un dominio (inglés) a secuencias en otro dominio (español). Es útil para la generación de texto como traducción automática y respuesta a preguntas. Definimos la arquitectura *Seq2Seq*. Estamos usando una capa de incrustación y una capa LSTM como codificador y otra capa LSTM seguida de una capa densa como decodificador.

# Ya entendemos cómo fluye el contexto a través de la red a través del estado oculto, elsiguiente paso es entender como fluye el contexto en ambas direcciones. Esto es lo que hace una **capa bidireccional**. Proporcionar un contexto futuro puede resultar en un mejor rendimiento del modelo. 
# 
# Esto puede parecer contrario a la forma en que los humanos procesan el lenguaje, ya que solo leemos en una dirección. Sin embargo, los humanos a menudo requieren un contexto futuro para interpretar lo que se dice. En otras palabras, a veces no entendemos una oración hasta que se proporciona una palabra o frase importante al final. Para implementar esto, entrenamos dos capas RNN simultáneamente. La primera capa se alimenta con la secuencia de entrada tal cual y la segunda se alimenta con una copia invertida.

# In[25]:


# Modelo LSTM
def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))    
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))

    return model


# In[26]:


# Modelo RNN
def build_model_rnn(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(SimpleRNN(units))
    model.add(RepeatVector(out_timesteps))    
    model.add(SimpleRNN(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))       
      
    return model


# In[27]:


# Modelo GRU
def build_model_gru(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(GRU(units, return_sequences=True))
    model.add(SimpleRNN(units))
    model.add(RepeatVector(out_timesteps))    
    model.add(Dense(out_vocab, activation='softmax'))
    
    return model


# Usamos el optimizador __'RMSprop (Root Mean Square Propagation)'__ ya que suele ser una buena opción para redes neuronales recurrentes. Otros tipos de optimizadores de redes neuronales profundas: Stochastic Gradient Descent (SGD), Adaptive Gradient Algorithm (AdaGrad), Adam (Adaptive moment estimation)...

# Como función de pérdida hemos utilizado __'sparse_categorical_crossentropy'__ porque nos permite usar la secuencia de destino tal y como está, en lugar de un formato codificado en caliente. Una codificación en caliente de las secuencias de destino con un vocabulario tan grande podría consumir toda la memoria de nuestro sistema. Otros tipos de funciones de pérdida: CategoricalCrossentropy, BinaryCrossentropy.

# Podemos comenzar a entrenar nuestro modelo. Lo entrenaremos durante 30 épocas, con un tamaño de lote de 512. También usaremos __ModelCheckpoint ()__ para guardar el mejor modelo con la menor pérdida de test. Este método sirve para evitar la parada anticipada.

# In[28]:


model = build_model(spa_vocab_size, eng_vocab_size, spa_length, eng_length, 512)
rms =   optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')


# In[29]:


model2 = build_model_rnn(spa_vocab_size, eng_vocab_size, spa_length, eng_length, 512)
rms =    optimizers.RMSprop(lr=0.001)
model2.compile(optimizer=rms, loss='sparse_categorical_crossentropy')


# In[30]:


model3 = build_model_gru(spa_vocab_size, eng_vocab_size, spa_length, eng_length, 512)
rms =    optimizers.RMSprop(lr=0.001)                                       # Defining the optimizing function
model3.compile(optimizer=rms, loss='sparse_categorical_crossentropy')       # Configuring the model for training


# A veces es útil apilar varias capas recurrentes una tras otra para aumentar el poder de representación de una red. Se deben obtener todas las capas intermedias para devolver secuencias completas.

# Más detalles sobre RepeatVector: https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/ Si se intenta forzar estas piezas del codificador-decodificador juntas, seobtiene un error que indica que la salida del decodificador es 2D y se requiere entrada 3D para el decodificador.
# 
# Podemos resolver esto usando una capa *RepeatVector* que simplemente repite la entrada 2D proporcionada varias veces para crear una salida 3D. La capa *RepeatVector* se puede utilizar como un adaptador para ajustar el codificador y el decodificador de la red juntos. Podemos configurarlo para que repita el vector de longitud fija en cada paso de tiempo en la secuencia de salida.
# 
# En resumen, el *RepeatVector* se utiliza como un **adaptador para ajustar la salida 2D de tamaño fijo del codificador a las diferentes longitudes y entradas 3D esperadas por el decodificador**. El contenedor *TimeDistributed* permite reutilizar la misma capa de salida para cada elemento en la secuencia de salida.

# In[ ]:


# DO NOT RUN IT (example of RepeatVector)
model = Sequential()
model.add(Dense(32, input_dim=32))

# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension
model.add(RepeatVector(3))

# now: model.output_shape == (None, 3, 32)


# En el ejemplo anterior, la capa *RepeatVector* repite las entradas entrantes un número específico de tiempo. La forma de la entrada anterior era (32,). Pero la forma de salida del *RepeatVector* fue (3, 32), ya que las entradas se repitieron 3 veces.

# In[31]:


print(spa_vocab_size)
print(eng_vocab_size)
print(spa_length)
print(eng_length)


# - **LSTM**

# La capa SimpleRNN no es bueno al procesar secuencias largas, como texto. Esto se debe al problema del gradiente de desaparición. Otros tipos de capas recurrentes funcionan mucho mejor como LSTM Y GRU.
# 
# **Long Short Term Memory o memoria a corto plazo (LSTM)** es una variante de la capa SimpleRNN que ya lleva información en muchos pasos de tiempo. Guarda la información para más tarde, evitando así que las señales más antiguas desaparezcan durante el procesamiento. Las unidades de memoria en un LSTM aprenden qué recordar y además pueden memorizar datos anteriores fácilmente.

# In[35]:


filename = 'model.h1.LSTM'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), 
          epochs=30, 
          batch_size=512, 
          validation_split = 0.2,
          callbacks=[checkpoint], verbose=1)


# In[29]:


model.summary()


# Comparamos la función de pérdida de train y test.

# In[30]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()


# - **RNN**

# **Recurrent Neural Networks o redes neuronales recurrentes (RNN)** son un tipo de red neuronal donde la salida del paso anterior se alimenta como entrada al paso actual. El estado oculto  recuerda alguna información sobre una secuencia. Los RNN tienen una “memoria” que recuerda TODA la información sobre lo calculado. Utiliza los mismos parámetros para cada entrada, ya que realiza la misma tarea en todas las entradas para producir la salida. Esto reduce la complejidad de los parámetros.

# In[28]:


filename = 'model.h1.RNN'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history2 = model2.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                     epochs=20,
                     batch_size=128,
                     validation_split=0.2)


# In[29]:


model2.summary()


# In[30]:


plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.legend(['train','validation'])
plt.show()


# - **GRU**

# **Gated recurrent units o unidades recurrentes cerradas (GRUs)** son un mecanismo de compuerta en las redes neuronales recurrentes. El GRU es como una memoria a corto plazo (LSTM) PERO con  menos parámetros. LSTM tiene tres puertas (entrada, salida y puerta de olvido) y GRU tiene dos puertas (puerta de reinicio y actualización). 
# 
# La puerta de actualización (z) ayuda al modelo a determinar cuánta información de los pasos de tiempo anteriores debe transmitirse al futuro. Mientras tanto, la puerta de reinicio (r) decide cuánta información pasada se debe olvidar. Hace que las RNN sean un poco más inteligentes. En lugar de permitir que toda la información del estado oculto fluya a través de la red, tenemos que ser más selectivos con la información sea más relevante. 
# 
# LSTM es más preciso en los conjuntos de datos que utilizan una secuencia más larga pero GRU usa menos parámetros de entrenamiento y por lo tanto usa menos memoria; se ejecuta más rápido que LSTM. En resumen, si la secuencia es grande o la precisión es muy crítica,es mejor usar LSTM pero si queremos una operación más rápida y que use menos memoria optaremos GRU.

# In[28]:


filename = 'model.h1.GRU'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history3 = model3.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), 
          epochs=30, 
          batch_size=512, 
          validation_split = 0.2,
          callbacks=[checkpoint], verbose=1)


# In[29]:


model3.summary()


# In[32]:


plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.legend(['train','validation'])
plt.show()


# Observando las funciones de pérdida de los tres modelos vemos que LSTM y GRU se sobre ajustan. Esto es algo normal ya que la traducción al final es muy dependiente del texto en el que está. Se comportan de forma muy parecida. 
# 
# LSTM (memoria a largo plazo) tiene tres puertas (entrada, salida y puerta de olvido), mientras que la GRU tiene dos puertas (puerta de reinicio y puerta de actualización). Las GRU asocian puertas de olvido y de entrada. Las GRU utilizan menos parámetros de entrenamiento y, por lo tanto, utilizan menos memoria, se ejecutan más rápidamente y se entrenan más rápido que las LSTM, mientras que éstas son más precisas en conjuntosde datos que utilizan una secuencia más larga (LSTM 80 min Y GRU 60 min).
# 
# En resumen, si la secuencia es grande o la precisión es muy importante, se empleanLSTMs, mientras que para un menor consumo de memoria y un funcionamiento más rápido se escogen GRUs. Cuanto menor sea la función de pérdida; mejor será el modelo. Por lo que el mejor modelo es LSTM ya se empieza a estabilizar antes. 

# ### Predicciones LSTM

# Cargamos el modelo guardado para hacer las predicciones.

# In[31]:


model = load_model('C:/Users/María/MASTER/Datos No Estruct/Texto/Practica/model.h1.LSTM')
preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))


# In[32]:


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


# In[33]:


# Convertir las predicciones en texto (inglés)
preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
             
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t)            
        
    preds_text.append(' '.join(temp))


# In[34]:


pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
pd.set_option('display.max_colwidth', 200)


# In[35]:


pred_df.head(15)


# In[36]:


pred_df.tail(15)


# In[37]:


pred_df.sample(15)


# ### Predicciones RNN

# In[31]:


model2 = load_model('C:/Users/María/MASTER/Datos No Estruct/Texto/Practica/model.h1.RNN')
preds = model2.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))


# In[32]:


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


# In[33]:


# Convertir las predicciones en texto (inglés)
preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
             
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t)            
        
    preds_text.append(' '.join(temp))


# In[34]:


pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
pd.set_option('display.max_colwidth', 200)


# In[35]:


pred_df.head(15)


# In[36]:


pred_df.tail(15)


# In[37]:


pred_df.sample(15)


# ### Predicciones GRU

# In[33]:


model3 = load_model('C:/Users/María/MASTER/Datos No Estruct/Texto/Practica/model.h1.GRU')
preds = model3.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))


# In[34]:


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


# In[35]:


# Convertir las predicciones en texto (inglés)
preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
             
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t)            
        
    preds_text.append(' '.join(temp))


# In[36]:


pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
pd.set_option('display.max_colwidth', 200)


# In[37]:


pred_df.head(15)


# In[38]:


pred_df.tail(15)


# In[39]:


pred_df.sample(15)


# Si comparamos a fondo las traducciones de los modelos; las traducciones de LSTM y GRU ya no se parecen tanto. 
