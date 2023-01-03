// Nolan Kyhl
// Fundamentals of Computing
// Lab 11 - Digit Recognition Neural Network - Function Definition File

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "projectfunc.h"

void OpenAndVerify(char filename[], FILE** file)
{ // Opens a file if it exists, or ends program if not.
	*file = fopen(filename, "r");
	if (*file == NULL) {
		printf("%s was not detected.\n", filename);
		exit(-1);
	} else {
		printf("%s opened successfully!\n", filename);
	}
}

void LoadImages(int NumImages, int ImageSize, Image Arr[], FILE **data)
{ // Populate Array of Image Structs with images from the desired dataset.
	char line[ImageSize*4];													// assumes up to 3 digits plus a comma for each pixel val.

	for (int iImage = 0; iImage < NumImages; iImage++)				
	{
		fgets(line, ImageSize*4, *data);									// Scan in Image line: 1 answer, 784 pixel values, seperated by commas
		Arr[iImage].answer = atoi(strtok(line, ","));

		for (int iPixel = 0; iPixel < ImageSize-1; iPixel++)		        // reads up until last pixel, converts 0-255 to 0-1
		{
			Arr[iImage].adjusted[iPixel] = (float) atoi(strtok(NULL, ",")) / 255;
		}
	
		Arr[iImage].adjusted[ImageSize-1] = (float) atoi(strtok(NULL, "\n")) / 255; // last pixel of line
	}
}

// ======================================

float rand_float()
{ // makes random float values between 0 and 1
	return (float)rand() / (float)RAND_MAX;
}

float Sigmoid(float x)
{ // "activation" function for neurons
	return 1 / (1 + exp(-x));
}

float dSigmoid(float x)
{ // derivative of Sigmoid
	return Sigmoid(x) * (1 - Sigmoid(x));
}

// ======================================

void InitializeWeights(int numLayer1, int numLayer2, float Weights[numLayer1][numLayer2], float DeltaWeights[numLayer1][numLayer2])
{ // Initializes the weights between two layers. Generalized to be used for setting up both Input->Hidden and Hidden->Output
	for (int i2 = 0; i2 < numLayer2; i2++)						// Loops through 2nd layer's neurons
	{
		for (int i1 = 0; i1 < numLayer1; i1++)					// Loops through previous layer's neurons to set weights (every connection between)
		{
			DeltaWeights[i1][i2] = 0.0;							// Delta weight will be used for adjusting weights later during backpropagation
			Weights[i1][i2] = 2*(rand_float() - 0.5);			// Sets connection between current neuron in layers 1 and 2 to a random val -1 to 1
		}
	}
}

void FeedImage(int IMAGESIZE, Image *img, float Input[])
{ // Copies current image's adjusted pixel values into the Input layer neuron's for use.
	for (int iPixel = 0; iPixel < IMAGESIZE; iPixel++)
	{
		Input[iPixel] = img->adjusted[iPixel];
	}
}

void ComputeNextLayer(int NumLayer1, int NumLayer2, float Weight[][NumLayer2], float Layer1[], float Layer2[])
{ // Computes next layer's neuron activations from current layer's values and connection weights.
	float Sum2[NumLayer2];	// Holds sum of previous layer's  contributions to each neuron in layer 2
	
	for (int i2 = 0; i2 < NumLayer2; i2++)				// Loop through neurons in current layer
	{
		Sum2[i2] = 0.0;										// keep track of current neuron's inputs

		for (int i1 = 0; i1 < NumLayer1; i1++)				// Loop through neurons in previous layer (inputs)
		{
			Sum2[i2] += Layer1[i1] * Weight[i1][i2];		// Adds contribution. Previous neuron*weighted connection
		}

		Layer2[i2] = Sigmoid(Sum2[i2]);					// Neuron's final value is Sigmoid of the sum we got.
	}
}

int maxNeuron(float Output[], int NumOutput)
{ // determines which Neuron has the highest value (confidence) and returns its index (corresponds to digit guess)
	float max = 0.0;
	int guess = -1;
	for (int iNeuron = 0; iNeuron < NumOutput; iNeuron++) {
		if (Output[iNeuron] >= max) {
			max = Output[iNeuron];
			guess = iNeuron;
		}
	}
	return guess;
}

// ======================================

void ShuffleInputs(Image arr[], int size)
{ // Takes in an image array and shuffles images
	if (size > 1) {
		for (int i = 0; i < size; i++) {							// loops through image indexes
			int j = i + rand() / (RAND_MAX / (size - i) + 1);		// gets random index after current
			Image Temp = arr[j];									// swaps current and random later image
			arr[j] = arr[i];
			arr[i] = Temp;
		}
	}
}

void PrintImage(Image arr[], int size, int imgIndex)
{ // Takes adjusted pixel values (0-1), multiplies by 255 to get back to original and displays.
	for (int iPixel = 0; iPixel < size; iPixel++)
	{
		if (iPixel % 28 == 0) printf("\n");
		printf("%3d ", (int) (arr[imgIndex].adjusted[iPixel]*255));
	}
	printf("\n");
}

void ClearImage(Image *img, int size)
{ // Sets all pixels to 0.
	for (int iPixel = 0; iPixel < size; iPixel++) {
		img->adjusted[iPixel] = 0.0;
	}
}
