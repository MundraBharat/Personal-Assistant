import numpy as np
import tensorflow as tf
import time


lines = open ('backup.txt', encoding = 'utf-8', errors= 'ignore').read().split('\n')


####remove timestamp
conversations = []
for line in lines:
    _line = line.split(':')
    if len(_line) == 4:
        conversations.append(_line[3])
    else:
        conversations.append(_line[0])
        
real_conversations= []    
for conv in conversations:
    _conv = conv.split('-')
    if _conv[0] == "me":
        _conv[0] = "BHARAT-"
        real_conversations.append(_conv[0]+_conv[1])
    elif _conv[0] == "Buuudddhiiii":
        _conv[0] = "ANINDITA-"
        real_conversations.append(_conv[0]+_conv[1])
    else:
        real_conversations.append(_conv[0])
   
      
count = 1
for conv in real_conversations:     
    if "-" not in conv:
        print(count, conv)
        real_conversations.remove(conv)
        #real_conversations.append("BHARAT- "+conv[0])
        #print (real_conversations
        count += 1


count = 1
for conv in real_conversations:     
    if "Sticker" in conv:
        print(count, conv)
        real_conversations.remove(conv)
        #real_conversations.append("BHARAT- "+conv[0])
        #print (real_conversations
        count += 1
        

count = 1
for conv in real_conversations:     
    if "Sent you a sticker" in conv:
        print(count, conv)
        real_conversations.remove(conv)
        #real_conversations.append("BHARAT- "+conv[0])
        #print (real_conversations
        count += 1
        

count = 1
for conv in real_conversations:     
    if "Nudge!" in conv:
        print(count, conv)
        real_conversations.remove(conv)
        #real_conversations.append("BHARAT- "+conv[0])
        #print (real_conversations
        count += 1
        


anindita = []
bharat = []

for conv in real_conversations:  
    if "ANINDITA-" in conv:
        anindita.append(conv)
    else:
        bharat.append(conv)

#final_conversations = []

questions = []
answers = []
for conv in anindita:
    questions.append(conv.replace("ANINDITA-",""))

for conv in bharat:
    answers.append(conv.replace("BHARAT-",""))

#test = {} 
#for key in range(len(anindita)): 
#    for value in anindita:
#        print("inside for")
#        test[key] = value 
#        anindita.remove(value)
#        break 


 
    
    
#Creatin dictionary that maps each word to its number of occurence
word2count = {}
for question in questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
    
#creating 2 dict that map ques words and answer words to unique integer
threshold = 1
questionsword2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionsword2int[word] = word_number
        word_number += 1
answersword2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answersword2int[word] = word_number
        word_number += 1   

#adding last tokens to these 2 dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionsword2int[token] = len(questionsword2int) + 1
for token in tokens:
    answersword2int[token] = len(answersword2int) + 1

#Creating INVERSE dictof d answersword2int dict
    ####isko or pdhna h chrome me link bookmark h
answersint2word = {w_i: w for w, w_i in answersword2int.items()}
    
#Adding the add of string token to the end of every answer
for i in range(len(answers)):
    answers[i] += ' <EOS>'
    
#translating all the ques and ans into inegers
# and replacing all the words that were filtered by <OUT>
#oxford dict ler sb words n number wise save kr l, pch biu e m use kr leyi hr word n number dena ko
question_into_int = []
for question in questions:
    ints = []
    for word in question.split():
        if word not in questionsword2int:
            ints.append(questionsword2int['<OUT>'])
        else:
            ints.append(questionsword2int[word])
    question_into_int.append(ints)
answer_into_int = []
for answer in answers:
    ints = []
    for word in answer.split():
        if word not in answersword2int:
            ints.append(answersword2int['<OUT>'])
        else:
            ints.append(answersword2int[word])
    answer_into_int.append(ints)
    
# Sorting ques and ans by the length of questions   --video 36 step 17
sorted_clean_questions = []
sorted_clean_answers = []    
for length in range(1, 12 + 1):
    for i in enumerate(question_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(question_into_int[i[0]])
            sorted_clean_answers.append(answer_into_int[i[0]])
            
################ Part 2 building seq2seq model
            
#creating placeholders for the inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None],name = 'input')   #(input type, dimension of array(for 2d matrix it is none,none), name given to input )
    targets = tf.placeholder(tf.int32, [None, None],name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')  #control d dropout rate
    return inputs, targets, lr, keep_prob
            
#Preprocessing the targets
def preprocess_targets(targets,word2int,batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])   #fill k andr matrix value h: jitne b batch honge unka 1st column or usme kya likhna h wo comma k bd 
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

#Creating the encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):  ## rnn size=number of input tensors
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    #added _, in starting because hme encoder_state chaiye, wo 2nd element hota h bidirectional_dynamic_rnn function ki
    #first vale h encoder_output <<   _, ki jgh encoder_output, b likh skte
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state

# decoding the training set
    # function k input parameters ko tensflow ki site pe pdhne ko milega embeddings wagerah and tf.variable.scope me b(decoding_scope wala)
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size]) #it is 3d array
    #attention keys for target states, values used for construct context(provided by encoder and used by decoder), score function used to identify similarties between keys and states
    #construct function used to build attention state
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    #if u dont want state and context state use <decoder_output, _, _ = tf...>
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

#Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    #inference=to deduce logically
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
    
# Creating the decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:  #TF scope: it contains several data entities
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

#Building seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionsword2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionsword2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionsword2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
        
############ part 3  trng d seq2seq model
# setting the hyperparameters
epochs = 10   #value of getting the batches of inputs into NN
batch_size = 50
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512    
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5  ##read doc "dropout: a simple way to prevent NN from overfitting"

#defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

#Loading d model inputs
inputs, targets, lr, keep_prob = model_inputs()

#Setting d seq length
sequence_length = tf.placeholder_with_default(5, None, name = 'sequence_length')  # used default fn as output is not fed into RNN

#Getting d shape f the input tensors
input_shape = tf.shape(inputs)

#Getting d trng n test predictions
#used tf.reverse fn to reshape d inputs   : reverse d dimension of tensors
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answersword2int),
                                                       len(questionsword2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionsword2int)

#setting up the loss err, d optimizer and gradient clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))#tf.ones is for weight
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

#Padding d seq with <PAD> token
# question: ['Who', 'are', 'you' ]      >>>                  ['Who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD> ]
#answer: [ <SOS>, 'I', 'am', 'a', 'bot' '.', <EOS> ]  >>>   [ <SOS>, 'I', 'am', 'a', 'bot' '.', <EOS>, <PAD> ] 
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting d data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionsword2int))     # o/p of apply padding is list, to work with TF we need  numpy array for this
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answersword2int))    
        yield padded_questions_in_batch, padded_answers_in_batch #yield is similar to return, used when work with sequences and inside loop
        
#splitting the ques n ans into trng and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
            
#Trng
batch_index_check_training_loss = 20 # we will check loss for evry 100 batches
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 500
checkpoint = "./chatbot_weights.ckpt" # for win users, replace this line by : checkpoint = "./chatbot_weights.ckpt"  otherwise remove ./
session.run(tf.global_variables_initializer()) # inside the run method, initialize all d tf global variables
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training loss error: {:>6.3f}, Training time on 100 batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print ('Validation loss error: {:>6.3f}, Batch Validation time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")


#######Part 4 - testing the seq2seq model

#Loading d weights n running d session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Converting d ques from strings to lists of encoding integers
def convert_string2int(question, word2int):
    #question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()] #get functn used if word exist in our dict then return int(word) otherwise return the int(<OUT>) from dict

#setting up d chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionsword2int)
    question = question + [questionsword2int['<PAD>']] * (5 - len(question))
    fake_batch = np.zeros((batch_size, 5))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersint2word[i] == 'i':
            token = 'I'
        elif answersint2word[i] == '<EOS>':
            token = '.'
        elif answersint2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersint2word[i]
        answer += token
        if token == '.':
            break
    print('Chotu: ' + answer)
    
            