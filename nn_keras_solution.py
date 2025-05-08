# Khanh Nguyen Cong Tran
# 1002046419

import tensorflow as tf
import numpy as np

# Let Keras use its default method for initialization of all weights (i.e., your code should not address this issue at all).
# For the optimizer, use "adam" with default settings.
# All hidden layers should be fully connected ("dense", in Keras terminology).
# The output layer should also be fully connected, and it should use the softmax activation function.
# The loss function should be Categorical Cross Entropy (it is up to you if you use the sparse version or not, 
#   but make sure that the class labels that you pass to the fit() method are appropriate for your choice of loss function).
# For any Keras option that is not explicitly discussed here 
#   (for example, the batch size), you should let Keras use its default values.
def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations): 

    #inital values
    input_shape = training_inputs[0].shape
    number_of_classes = np.max(training_labels) + 1

    # create the model 
    model = tf.keras.Sequential()

    # add the input layer
    model.add(tf.keras.layers.Input(shape = input_shape))

    # add the hidden layers
    for i in range(layers - 2): 
        model.add(tf.keras.layers.Dense(units_per_layer[i], activation=hidden_activations[i]))

    # add the output layer
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    # compile
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])
    
    #training
    model.fit(training_inputs, training_labels, epochs=epochs)

    return model

    

def test_model(model, test_inputs, test_labels, ints_to_labels): 
    #keras built in evaluation function
    # test_loss, test_acc = model.evaluate(test_inputs,  test_labels, verbose=0)

    # this is manual testing
    correct = 0
    length = test_inputs.shape[0]
    for i in range(length): 
        test_data = np.expand_dims(test_inputs[i], axis=0)
        test_label = test_labels[i, 0]

        prediction_scores = model.predict(test_data)

        predicted = ints_to_labels[np.argmax(prediction_scores)]
        true_class = ints_to_labels[test_label]
        
        accuracy = test_output(predicted_scores=prediction_scores, correct_class=test_label)
        correct += accuracy
        print_test_object(object_id=i, predicted_class=predicted, true_class=true_class, accuracy=accuracy)
    
    classification_accuracy = correct / length
    # print_classification_accuracy(classification_accuracy=classification_accuracy)

    return classification_accuracy


# print test objects
def print_test_object(object_id: int, predicted_class: any, true_class: any, accuracy: float): 
    print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % 
           (object_id, str(predicted_class), str(true_class), accuracy))
def print_classification_accuracy(classification_accuracy: float) -> None: 
    print('MANUAL classification accuracy=%6.4f\n' % (classification_accuracy))

    
def test_output(predicted_scores: any, correct_class: int) -> float: 
    predicted_scores = np.squeeze(predicted_scores) 

    maximum_score = np.max(predicted_scores)

    tied_classes = np.where(predicted_scores == maximum_score)[0]

    # if we only get one maximum value then check if its correct of not
    if len(tied_classes) == 1: 
        predicted_class = tied_classes[0]
        if predicted_class == correct_class: 
            return 1
    # we got ties 
    else: 
        # if the correct class is withiin the ties -> divided by the number of ties
        if correct_class in tied_classes: 
            return 1.0 / len(tied_classes)

    # all ifs fail -> it is wrong then 
    return 0