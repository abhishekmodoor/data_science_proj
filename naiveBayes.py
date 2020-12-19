# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import heapq

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    """
    Creating BlankImageCount
    """
    blankImageCount = {}
    for feature in self.features:
      blankImageCount[feature] = 0
    """
    Creating Counts
    """
    counts = {}
    occurancesOfLabel = {}
    for label in self.legalLabels:
      occurancesOfLabel[label] = 0
      counts[label] = blankImageCount.copy()
    """
    Counting...
    """
    for index, image in enumerate(trainingData):
      onePixels = [k for k,v in image.iteritems() if v == 1]
      label = trainingLabels[index]
      occurancesOfLabel[label] += 1
      for pixel in onePixels:
        counts[label][pixel] += 1

    """
    Calculating Prior Distribution
    """
    probabilityOfY = {}
    n = len(trainingData)
    for index, label in enumerate(occurancesOfLabel):
      probabilityOfY[label] = float(occurancesOfLabel[label]) / n
    self.probabilityOfY = probabilityOfY
    """
    Calculating Conditional Probabilities & best k
    """
    bestK = -1
    bestKCorrectness = -1
    bestConditionalProbability = None
    numValidationData = len(validationData)
    for k in kgrid:
      print("Trianing k =",k,"...")
      totalRight = -1
      """
      calculate conditional probabilities with smoothing k 
      """
      conditionalPixelProbalities = counts.copy()
      for label in conditionalPixelProbalities:
        for pixel in conditionalPixelProbalities[label]:
          conditionalPixelProbalities[label][pixel] = (float(conditionalPixelProbalities[label][pixel]) + k) / (occurancesOfLabel[label] + (2 * k)) 
      """
      evalulate on validation data
      """
      for index, image in enumerate(validationData):
        """
        Find the best label 
        """
        overallProbablility = 0
        bestMatch = -1
        for label in conditionalPixelProbalities:
          # testing current label on current image
          temporaryProbability = probabilityOfY[label]
          #print(temporaryProbability)
          for pixel in image:
            if validationData[label][pixel] == 1:
              # this pixel is 1
              temporaryProbability *= conditionalPixelProbalities[label][pixel]
            else: 
              # this pixel is 0
              temporaryProbability *= 1 - conditionalPixelProbalities[label][pixel]
          if temporaryProbability > overallProbablility:
            overallProbablility = temporaryProbability
            bestMatch = label 
        if bestMatch == validationLabels[index]:
          totalRight += 1
      thisKCorrectness = float(totalRight) / numValidationData
      if thisKCorrectness > bestKCorrectness:
        bestKCorrectness = thisKCorrectness
        bestK = k
        bestConditionalProbability = conditionalPixelProbalities
    self.conditionalProbabilities = bestConditionalProbability

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    for label in self.conditionalProbabilities:
        # testing current label on current image
        temporaryProbability =  math.log(self.probabilityOfY[label])
        for pixel in datum:
          if datum[pixel] == 1:
            # this pixel is 1
            temporaryProbability +=  math.log(self.conditionalProbabilities[label][pixel])
          else: 
            # this pixel is 0
            temporaryProbability +=  math.log(1 - self.conditionalProbabilities[label][pixel])
        logJoint.incrementAll([label], temporaryProbability)
    
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    heap = []
    for feature in self.features:
      odds = self.conditionalProbabilities[1][feature]/self.conditionalProbabilities[0][feature]
      if len(heap) < 100:
        heapq.heappush(heap, (odds,feature))
      elif odds > heap[0]:
        heapq.heappushpop(heap, (odds,feature))
    featuresOdds = [x[1] for x in heap]

    return featuresOdds
    

    
      
