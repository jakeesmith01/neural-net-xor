#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double sigmoid(double x){
  if(x > 20) return 1.0;
  if(x < -20) return 0.0;

  double z = exp(-x);
  return 1.0 / (1.0 + z);
}
double dSigmoid(double x){
  return x * (1.0 - x);
}

double init_weights(){ 
  double placeholder = ((double)rand()) / ((double)RAND_MAX);
  return placeholder;
}

void shuffle(int *array, size_t n){
  if(n > 1){
    size_t i;
    for(i = 0; i < n - 1; i++){
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 1
#define NUM_TRAINING_SETS 4

int main(){
  //learning rate 
  const double lr = 0.1f;

  double hiddenLayer[NUM_HIDDEN];
  double outputLayer[NUM_OUTPUTS];

  double hiddenLayerBias[NUM_HIDDEN];
  double outputLayerBias[NUM_OUTPUTS];
  
  double hiddenWeights[NUM_INPUTS][NUM_HIDDEN];
  double outputWeights[NUM_HIDDEN][NUM_OUTPUTS];

  double trainingInputs[NUM_TRAINING_SETS][NUM_INPUTS] = {{0.0f, 0.0f}, {1.0f, 0.0f},
                                                          {0.0f, 1.0f}, {1.0f, 1.0f}};

  double trainingOutputs[NUM_TRAINING_SETS][NUM_OUTPUTS] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

  for(int i = 0; i < NUM_INPUTS; i++){
    for(int j = 0; j < NUM_HIDDEN; j++){
      hiddenWeights[i][j] = init_weights();
    }
  }

  for(int i = 0; i < NUM_HIDDEN; i++){
    for(int j = 0; j < NUM_OUTPUTS; j++){
      outputWeights[i][j] = init_weights();
    }
  }

  for(int i = 0; i < NUM_OUTPUTS; i++){
    outputLayerBias[i] = init_weights();
  }

  for(int i = 0; i < NUM_HIDDEN; i++){
    hiddenLayerBias[i] = init_weights();
  }

  int trainingSetOrder[] = {0,1,2,3};

  int numEpochs = 10000;

  //training loop
  for(int epoch = 0; epoch < numEpochs; epoch++){
    shuffle(trainingSetOrder, NUM_TRAINING_SETS);

    for(int x = 0; x < NUM_TRAINING_SETS; x++){
      int i = trainingSetOrder[x];

      //forward pass
      //compute hidden layer activation 
      for(int j = 0; j < NUM_HIDDEN; j++){
        double activation = hiddenLayerBias[j];

        for(int k = 0; k < NUM_INPUTS; k++){
          activation += trainingInputs[i][k] * hiddenWeights[k][j];
        }
        //printf("Hidden layer activation[%d] = %f\n", j, activation);
        hiddenLayer[j] = sigmoid(activation);

        if(isnan(hiddenLayer[j])){
          //printf("NaN in hidden layer at index %d\n", j);
        }
      }

      //compute output layer activation 
      for(int j = 0; j < NUM_OUTPUTS; j++){
        double activation = outputLayerBias[j];
        for(int k = 0; k < NUM_HIDDEN; k++){
          activation += hiddenLayer[k] * outputWeights[k][j];
        }
        //printf("Output layer activation[%d] = %f\n", j, activation);
        outputLayer[j] = sigmoid(activation);
        if(isnan(outputLayer[j])){
          //printf("NaN in output layer at index %d\n", j);
        }
      }

      printf("Input: %g %g    Output: %g    Predicted Output: %g\n", trainingInputs[i][0], trainingInputs[i][1], outputLayer[0], trainingOutputs[i][0]);
      
      //Back prop
      //compute change in output weights 
      
      double deltaOutput[NUM_OUTPUTS];

      for(int j = 0; j < NUM_OUTPUTS; j++){
        double error = (trainingOutputs[i][j] - outputLayer[j]);
        deltaOutput[j] = error * dSigmoid(outputLayer[j]);
        if(isnan(deltaOutput[j])){
          //printf("NaN in deltaOutput at index %d\n", j);
        }
      }
      
      //compute change in hidden weights 

      double deltaHidden[NUM_HIDDEN];

      for(int j = 0; j < NUM_HIDDEN; j++){
        double error = 0.0f;
        for(int k = 0; k < NUM_OUTPUTS; k++){
          error += deltaOutput[k] * outputWeights[j][k];
        }
        deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
        if(isnan(deltaHidden[j])){
          //printf("NaN in deltaHidden at index %d\n", j);
        }
      }

      //apply change in output weights 
      for(int j = 0; j < NUM_OUTPUTS; j++){
        outputLayerBias[j] += deltaOutput[j] * lr;

        for(int k = 0; k < NUM_HIDDEN; k++){
          outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
          if(isnan(outputWeights[k][j])){
            //printf("NaN in outputWeights at [%d][%d]\n", k, j);
          }
        }
      }

      for(int j = 0; j < NUM_HIDDEN; j++){
        hiddenLayerBias[j] += deltaHidden[j] * lr;
        for(int k = 0; k < NUM_INPUTS; k++){
          hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * lr;
          if(isnan(hiddenWeights[k][j])){
            //printf("NaN in hiddenWeights at [%d][%d]\n", k, j);
          }
        }
      }
    }
      
     
    


      
      
    





  }
  //print final weights after training 
      printf("\nFinal Hidden Weights: \n");
      for(int j = 0; j < NUM_HIDDEN; j++){
        for(int k = 0; k < NUM_INPUTS; k++){
          printf("%f ", hiddenWeights[k][j]);
        }
      }

      printf("\n");

      printf("Final Hidden Biases: \n");
      for(int j = 0; j < NUM_HIDDEN; j++){
        printf("%f ", hiddenLayerBias[j]);
      }

      printf("\n\n\n");
      
      printf("Final Output Weights: \n");
      for(int j = 0; j < NUM_OUTPUTS; j++){
        for(int k = 0; k < NUM_HIDDEN; k++){
          printf("%f ", outputWeights[k][j]);
        }
      }

      printf("\n");


      printf("Final Output Biases: \n");
      for(int j = 0; j < NUM_OUTPUTS; j++){
        printf("%f ", outputLayerBias[j]);
      }
 
  return 0;
  
}


