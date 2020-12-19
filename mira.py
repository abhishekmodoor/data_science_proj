# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
from collections import Counter
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = .001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.3,0.03,0.003,0.0003,0.00003,0.6,0.06,0.006,0.0006,-0.3,-0.03,-0.003,-0.0003,-0.00003,-0.6,-0.06,-0.006,-0.0006]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    "*** YOUR CODE HERE ***"

    #if predicted label is not equal to actual label
    num_errors = 0 
    
    #weights will be changed when checking if labels are equal to each other
    

    
    #traversing across the Cgrid to train each set across each value of c in Cgrid 
    for c in Cgrid:
      updatedWeights = self.weights.copy()
      for iteration in range(self.max_iterations):
      
        print("Starting iteration ", iteration, "..")
        if iteration > 0:
         num_errors = 0

        for i in range(len(trainingData)):
          trainingUnit = trainingData[i].copy() #trainingUnit is one instance of training data at i
          #predLabel = self.classify(trainingUnit) #classifies data in order list of predicted label values
          #predictedLabel = predLabel[0] #extract predicted label where max is at first index
          realLabel = trainingLabels[i] #extract real label from training label in order to compare



          predY = 0
          predictedLabel = -1;
          for label in self.legalLabels:
           predLabel = trainingUnit * updatedWeights[label]
           if predictedLabel < predLabel or predictedLabel == -1:
             predictedLabel = predLabel
             predY = label

          tau = 0  
          
          #if predicted label is not equal to real label
          if predY != realLabel: 
           feature = trainingUnit.copy() #extract feature of current training unit
           num_errors += 1 
                            #t = ((wpred - wactual)*feature + 1.0)/(2 * feature * feature) = num/div      
           num = updatedWeights[predY] - updatedWeights[realLabel]
           num = num * feature
           num += 1.0   
           

           div = (feature*feature)
           
           div += 2.0
           t = num/div
           
           tau = min(c,t)
           
           
           
           #for j in range(feature):
           for j in range(len(trainingData[i])):
            feature[j] = feature[j] * tau
           updatedWeights[realLabel] = updatedWeights[realLabel] + feature #wactual = wactual + tau*feature
           updatedWeights[predY] = updatedWeights[predY] - feature #wpred = wpred + tau*feature
            

      print("finished updating weights")

    #determine guesses by classifying validation data
    guesses = self.classify(validationData)
    correct = 0
    bestAccuracy = None #no best accuracy rate yet

    #traverse over guesses, determine how many 
    #answers were correct 
    for i in range(len(guesses)):
      if guesses[i] == validationLabels[i]: #guess matches validation label
        correct += 1

    accuracy = correct / len(guesses) #determine percentage
    if(accuracy > bestAccuracy):
      bestAccuracy = accuracy

    self.weights = updatedWeights
        
  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    return featuresOdds

