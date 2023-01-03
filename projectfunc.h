// Nolan Kyhl
// Fundamentals of Computing
// Lab 11 - Digit Recognition Neural Network - Header File

typedef struct {
	int answer;					// the digit label for the image
	float adjusted[784];		// 28 x 28 pixels flattened = 784 vals. Adjusted to be 0-1
} Image;

void OpenAndVerify(char [], FILE**);
void LoadImages(int, int, Image [], FILE**);

float rand_float();
float Sigmoid(float);
float dSigmoid(float);

void InitializeWeights(int n1, int n2, float [n1][n2], float [n1][n2]);
void FeedImage(int, Image *, float []);
void ComputeNextLayer(int n1, int n2, float [n1][n2], float [], float []); 
int maxNeuron(float [], int);

void ShuffleInputs(Image [], int);
void PrintImage(Image [], int, int);
void ClearImage(Image *, int);
