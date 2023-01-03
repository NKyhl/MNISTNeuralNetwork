// Nolan Kyhl
// Fundamentals of Computing
// Lab 11 - Digit Recognition Neural Network - Main file

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "projectfunc.h"

int main()
{
	srand(time(0));

	// Open Data Files
	FILE *traindata;
	FILE *testdata;

	OpenAndVerify("mnist_train.csv", &traindata);
	OpenAndVerify("mnist_test.csv", &testdata);

	// Read in Images
	int TRAINSIZE = 2000;  								// # of training images to use - can be up to 60,000 as memory allows
	int TESTSIZE = 100;  								// # of testing images to use - can be up to 10,000 as memory allows
	int IMAGESIZE = 784;  								// 28 pixels x 28 pixels

	Image trainArr[TRAINSIZE];							// create array of image structs to hold training data
	Image testArr[TESTSIZE];							//								''					  testing data

	LoadImages(TRAINSIZE, IMAGESIZE, trainArr, &traindata);	// Populate all training structs with training images.
	LoadImages(TESTSIZE, IMAGESIZE, testArr, &testdata);	// Populate all testing structs with testing images.

	fclose(traindata);
	fclose(testdata);

	// ===== PART ONE: TRAIN THE NETWORK =====
	// For this stage, I will leave the "training" without many functions so you can see the whole process in front of you.
	//	For Parts Two and Three, the work will be done within functions.

	// Set up Network (Layers and Weights)
	int NumInput = IMAGESIZE;							// corresponds to an image's pixels being fed into network (each is a neuron)
	int NumHidden = 100;								// hidden layer neurons. Somewhat arbitrary number
	int NumOutput = 10;									// each output neuron corresponds to a digit 0-9. The network's guess

	float Input[NumInput];								// Holds Input layer's activation values
	float Hidden[NumHidden];							//	Holds Hidden layer's activation values
	float Output[NumOutput];							//	Holds Output layer's activation values

	float WeightIH[NumInput][NumHidden];			// Each connection between input and hidden neurons has a unique weight
	float WeightHO[NumHidden][NumOutput];			// Same as above for Output layer, and it's connections with the hidden layer

	// Training Stats and Parameters
	float Error;										// How much each Output neurons were off from the ideal, squared and summed up over epoch for avg.
	int numCorrect;										// How many images were correctly identified this epoch.
	int guess;											// Network's guess for training images.
	float alpha = 0.2;									// Learning rate: previous neuron's contribution to the weight nudge.
	float beta = 0.2;									// Learning rate: nudge from last epoch's contribution (like momentum)
	int TotalEpochs = 50;								// Total number of times my training set will be shown to the network.

	// Useful Arrays for Feeding Forward 
	float SumH[NumHidden];								//	Holds the sum of each hidden neuron's inputs (all the connections coming into the neuron)
	float SumO[NumOutput];								//	Holds the sum of each output neruon's inputs	(all the connections coming into the neuron)

	// Useful Arrays for Backpropagation
	float DeltaO[NumOutput];							// change in each Output neuron's activation value. Not the weight "connection" with another neuron!
	float DeltaH[NumHidden];							// 		...		Hidden		...

	float DeltaWeightHO[NumHidden][NumOutput];	        //	change in each H->O weight for tweaking it and teaching the network
	float DeltaWeightIH[NumInput][NumHidden];		    // 		...		I->H			...

	float SumDOW[NumHidden];							// Holds sum of weight changes from a hidden neuron to all output neurons (Delta Output Weight)
														//		- for calculating the change in Hidden layer neuron values, propagating error backwards
 
	printf("Training the the network over %d epochs (~90 seconds for 50): \n", TotalEpochs);

	InitializeWeights(NumInput, NumHidden, WeightIH, DeltaWeightIH);	// Set random connection weights between Input and Hidden Layers
	InitializeWeights(NumHidden, NumOutput, WeightHO, DeltaWeightHO);	// Set random connection weights between Hidden and Output Layer
			
	for (int epoch = 1; epoch <= TotalEpochs; epoch++) 				    // Repeatedly inputs the training data, adjusting weights after each image.
	{
		Error = 0.0; 													// "Error" variable keeps a running sum of the epoch's error.
		numCorrect = 0;
		guess = -1;
		ShuffleInputs(trainArr, TRAINSIZE);								// Shuffles images to prevent network from learning their order. 

		for (int iImage = 0; iImage < TRAINSIZE; iImage++)				// Loop through training images
		{	
			FeedImage(IMAGESIZE, &trainArr[iImage], Input);				// Feeds current image into the Input layer
		
			// Compute Hidden Layer Activations from Input
			for (int i2 = 0; i2 < NumHidden; i2++)							// Loops through neurons in the Hidden layer
			{
				SumH[i2] = 0.0;												// Holds sum of all current neuron's inputs
		
				for (int i1 = 0; i1 < NumInput; i1++)						// Loops through all neurons in the previous layer (Input)
				{
					SumH[i2] += Input[i1] * WeightIH[i1][i2];				// Adds contribution. Input Neuron * weighted connection
				}
		
				Hidden[i2] = Sigmoid(SumH[i2]);								// Neuron's activation is the sigmoid of that sum.
			}
	
			// Compute Output Layer Activations from Hidden layer
			for (int i3 = 0; i3 < NumOutput; i3++)							// (Same as above)
			{
				SumO[i3] = 0.0;
		
				for (int i2 = 0; i2 < NumHidden; i2++)
				{
					SumO[i3] += Hidden[i2] * WeightHO[i2][i3]; 
				}
			
				Output[i3] = Sigmoid(SumO[i3]);

				// Calculate Neuron's Error			
				if (i3 == trainArr[iImage].answer) {						// The correct neuron should be 1.0
					Error += pow(Output[i3]-1.0, 2);					    //      Error for running tally
					DeltaO[i3] = (Output[i3]-1.0)*dSigmoid(SumO[i3]);	    //      Calculate the nudge 
				} else {													// All other neurons should be 0.0
					Error += pow(Output[i3]-0.0,2);							//		' '
					DeltaO[i3] = (Output[i3]-0.0)*dSigmoid(SumO[i3]);	    //		' '
				}
			}

			// Determine Network's Guess for this image
			guess = maxNeuron(Output, NumOutput);
			if (guess == trainArr[iImage].answer) numCorrect++;

			// Update Hidden -> Output weights
			for (int i3 = 0; i3 < NumOutput; i3++)
			{
				for (int i2 = 0; i2 < NumHidden; i2++)
				{
					DeltaWeightHO[i2][i3] = alpha*Hidden[i2]*DeltaO[i3] + beta*DeltaWeightHO[i2][i3];   // Current value's contribution + "momentum"
					WeightHO[i2][i3] -= DeltaWeightHO[i2][i3];										    // nudge weight
				}
			}

			// Backpropagate Error to Hidden layer
			for (int i2 = 0; i2 < NumHidden; i2++)
			{
				SumDOW[i2] = 0.0;										    // holds sum of "DOW" - delta output weights for this neuron.
				for (int i3 = 0; i3 < NumOutput; i3++)
				{
					SumDOW[i2] += WeightHO[i2][i3] * DeltaO[i3];
				}
				DeltaH[i2] = SumDOW[i2] * (Hidden[i2]*(1.0 - Hidden[i2]));	// Calculate Hidden neuron's nudge from weight changes and derivative
			}

			// Update Input -> Hidden weights
			for (int i2 = 0; i2 < NumHidden; i2++)
			{
				for (int i1 = 0; i1 < NumInput; i1++)
				{
					DeltaWeightIH[i1][i2] = alpha*Input[i1]*DeltaH[i2] + beta*DeltaWeightIH[i1][i2];	// (See above)
					WeightIH[i1][i2] -= DeltaWeightIH[i1][i2];
				}
			}
		}

		printf("  Epoch %-2d - Accuracy: %5.3g - Avg Error: %g\n", epoch, (float)numCorrect/ TRAINSIZE, Error/TRAINSIZE);
	}		
	
	printf("Network Trained! Final Accuracy: %d/%d images correct from the Train set.\n", numCorrect, TRAINSIZE);
	printf("====================================================\n");

	// ===== PART TWO: TEST THE NETWORK =====

	printf("Testing Network with first %d images from the Test Dataset.\n", TESTSIZE);
	int dispIncorrect;
	printf("Would you like to display the incorrect images after test? (1/0): ");
	scanf("%d", &dispIncorrect);
	printf("Training...\n");
	
	// Reset stats
	numCorrect = 0;
	guess = -1;

	// Feed Testing images through network. 
	for (int iImage = 0; iImage < TESTSIZE; iImage++)
	{
		FeedImage(IMAGESIZE, &testArr[iImage], Input);
		ComputeNextLayer(NumInput, NumHidden, WeightIH, Input, Hidden);	// calculates Hidden neuron activations
		ComputeNextLayer(NumHidden, NumOutput, WeightHO, Hidden, Output);	// calculate Output neuron activations.
		guess = maxNeuron(Output, NumOutput);

		//	Keep track of # of correct guesses.
		if (guess == testArr[iImage].answer) {
			 numCorrect++;
		// Display incorrect guesses to the user, if asked.
		} else if (dispIncorrect) {
				printf("INCORRECTLY GUESSED IMAGE BELOW - LABEL: %d - GUESS %d.\n", testArr[iImage].answer, guess);
				PrintImage(testArr, IMAGESIZE, iImage);
		}
	}

	printf("The network got %d/%d test images correct. %g percent accurate.\n", numCorrect, TESTSIZE, (float)numCorrect/(float)TESTSIZE);
	printf("====================================================\n");
	
	// ===== PART THREE: USER DRAWING NOT AVAILABLE THROUGH GITHUB =====
	return 0;
}
