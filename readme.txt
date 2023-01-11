Nolan Kyhl
Fundamentals of Computing
Lab 11 - Digit Recognition Neural Network - README

= This is long, but I wanted to try to explain it so you have an idea and so I can work through the process =

CONTEXT:
	This semester I've become really interested in machine learning. I watched a great video on the 3Blue1Brown
Youtube channel about neural networks and how they work, and it blew my mind. How can a computer "learn"??
Turns out it's just a bunch of multiplication and addition when it really comes down to it. While after
watching those couple videos I had gotten a good idea about how neural networks work in the big picture,
I still didn't feel like I knew the nitty gritty details enough, and I wanted to try to create one from
scratch, which would require me to understand it more fully. I've linked the 3Blue1Brown playlist below
if you are curious, or want some background on what is going on. It's really well made and animated:

	https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

=============================================================================================================

WHAT IS IT:
	This program is a neural network whose purpose is to identify handwritten digits. When you run the program,
 it will take a full minute or two to train the network with a dataset of images of handwritten digits. After
50 "epochs" (will be mentioned later) the network should be doing pretty well, so it ends the training phase.
Then the program will begin to test the network, by giving it a set of new images (100) that it's never seen
before. Before doing so, it asks you if you want to display the images it gets incorrect. It's an interesting
feature to see how it mislabels the numbers (ex: calls an 8 a 3... pretty similar!). Feel free to disregard
and enter 0 (no) if you don't want all that output. It will then pretty instantaneously display its accuracy
on the test images (that it's never seen). Usually not as good as it is on the training set, which it was
trained to identify. 
	The last part is that a graphics window pops up for you to draw your own digit, and have the network try
to identify it. It's not perfect, but for a relatively simple network and a relatively quick period of time
training it, it's pretty cool. Some tips on drawing digits:
		1) Draw your digit big. Fill up much of the space, because that's what the training images are like.
		2) Make it look handwritten. It's trained on curvy, handwritten digits, so that's what it recognizes best
		3) Center it. Otherwise for example, a '1' will start to look like the stem of a '4'.
		4) After drawing, hit 'SPACEBAR' to send it into the network and receive a guess.
	Again, it's not perfect. I've had terrible luck trying to get it to recognize sixes, and it gets 3's and 8's
often confused as well as 2's and 7's (understandably). But otherwise it's pretty good!

=============================================================================================================

DATABASE SOURCE:
	Dariel Dato-On adapted the widely used MNIST database of hand-drawn digits into a csv format on Kaggle.
There are two seperate files: Firstly, mnist_train.csv that has 60,000 training images and labels, and then
mnist_test.csv which contains 10,000 test examples. (I really just scratch the surface of these files).
Each image is a row in the file, and each pixel is seperated by commas. The first number is it's label.
Each image is 28 pixels by 28 pixels, grayscale, with a black background (0) and white numbers (up to 255).

		https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

=============================================================================================================

HOW DOES IT WORK:
	After opening the files, a portion of the data is read in (Default: 2000 training images, 100 test images).
My main data structure is an array of "Image" structs, with each struct having an "answer" attribute, which
is the image's label, and a "adjusted" attribute (a 1x784 float array). This array holds a flattened version
of the image's pixel values (28 x 28 = 784) that have been adjusted to range from 0-1 (instead of 0-255) by
dividing by 255.

= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	
PART ONE: TRAINING THE NETWORK
	(Googling "neural network" and looking at a picture can help visualize). My neural network consists of 
three layers of neurons. Layer 1 is the input layer, which simply just holds the adjusted pixel values of an 
image that I put in, thus it consists of 784 neurons, one for each pixel. A neuron can be thought of as a box 
that just holds a value.
	Layer 2 is the "hidden" layer, called so because you don't really see what's happening in it while using 
the network. It consists of 100 neurons, which is a pretty arbitrary number. Increasing it could help the network
identify traits about digits better, but it also makes the training slower.
	Layer 3 is the "output" layer, which outputs the network's guess of digit. Thus, it has 10 neurons. One for
each of the digits we want to recognize (0->9). Ex: The decimal that the 3rd output neuron holds is it's confidence
that the image is a 3!
	The last important part of the network is that each neuron in one layer is connected to EACH neuron in the
next layer, which makes the shape you see when you google "neural network" Each connection is has what is called
a weight, which is just a float from -1 to 1. To calculate a neuron in the hidden layer's value, you take the sum
of all the contributions of the neurons in the previous layer (all the lines coming in), by multiplying each 
neuron by the weight. Repeat for all the hidden layer neurons, and then do the same thing from the hidden layer
to the output. Normally, each neuron has a bias, which shifts it's value and can help the network learn, but
it's doing well enough currently that I'll add that in the future.
	Each Neuron's value is held in a 1D array, called Input[], Hidden[], or Output[].
	Each weight is held in a 2D array connecting two layers, called WeightIH[][], and WeightHO[][].

===

	How does it learn then? I first take a training image, place it's values in the input layer, and then multiply
and add through, getting the values for the hidden layer, and then finally the output layer (with a bunch of for
loops). This is called "feeding the image forward". Only problem is, our output is going to be terrible, because
to begin with, I assign all weights a random float value from -1 to 1. Of course it's not going to work. What you
do though, is you calculate the error for that image. I want the Output Layer to look like this for a "2" image:

		Neuron #:		0	1	2	3	4	5	6	7	8	9
		Ideal Output:		0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0

	So I subtract how much each neuron was off from what it should be, sum all that up, and I've got my error. I
also square it, which makes sure it's positive and also just amplifies it so it has a greater effect.
	
	The real magic of the neural network is what happens next: "backpropagation." After I've fed an image through
the network and calculated how much it was off by, I then take that error, and go backwards through the network,
tweaking each weight slightly in the right direction. For example, if I got 0.4 for the 2nd neuron and it should
be 1.0, I'm going to tweak each weight coming into that neuron in order for the result to be higher. When a
neuron's value is calculated, the sum of the previous connections is put in a function called Sigmoid, which
is commonly used. It means that slight tweaks will cause substantial changes in the output. I still don't fully
grasp why this one is chosen and why it's needed, I just know generally that we need to "activate" a neuron using
an "activation function" like Sigmoid.
	Conceptually, if you keep doing this, (feeding an image in, calculating how much it was off by, and adjusting
the weights to get closer), you'll eventually get to a place where feeding in an image's pixels will output just 
the right values for it to tell you what digit it is. That's how it's "learning" and how we "train" it. We just
make estimated nudges to each, and do it a bunch of times, which computers are good at.

	What makes an estimated "nudge"? When you think back to high school math, we often found the minimum of a 
function, which is exactly what's happening here (the minimum of an error function), just in higher dimensions.
If you took a normal y(x) function and were given a random point on the curve without knowing the function, you 
could get closer to a minimum by finding the derivative at that point, and shifting in that direction. The
steeper the slope, the bigger jump you could make to get to the bottom. That's why DeltaO[] (holds the shifts
to be done on the Hidden->Output weights) is calculated using dSigmoid. Sigmoid has a clean derivative:
dSigmoid = sigmoid * (1-sigmoid), which is a reason why it's used.
	Another factor is that if a neuron in the previous layer has a BIG value, a small adjustment in the weights
coming out of it will have a BIG change in the output. This is taken into account when that DeltaO[] from before
is multiplied by Hidden[]. Depending on the magnitude of the neuron, you should focus more on that one - shift
it more because it's more important to changing the output.
	Lastly, you'll see float variables "alpha" and "beta." These affect the learning rate, by adjusting how much
the above-mentioned factors should nudge the weights. Setting them high will make the weights jump around a bunch,
and might make the learning process faster, but will make it hard for the program to converge on a real minimum.
It might get stuck on a local min, which is what you see when the accuracy during training stays on one value for
a long time. Sometimes it'll get out of the rut and jump down again, sometimes not. Setting these too low will make
the shifts so miniscule that the network takes a while to train. I've found that 0.2 for both works pretty well

	During training, an "epoch" is me throwing all 2000 training images I've loaded in at the network. Each time
the weights are shifted, and then I go again, putting all 2000 through. One way I could improve the speed is
called "stochastic" gradient descent, where I through smaller random batches at the network. It's not as accurate
though since you're not showing it all the examples every time, and I don't fully get it yet, so I decided not to 
try it, at least now.

= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

PARTS TWO & THREE: TESTING THE NETWORK
	Now the network is "trained" reasonably (it's weights are such that when you put in an image, it spews out the
right kind of output at the end - one that tells us what digit is in the image. Usually on the training set, it'll
get around 92-98% accuracy. Now to really test out whether it's "learned" anything, I give it 100 images from the
test dataset, "mnist_test.csv" that it's never seen before. If I show it a "2", maybe it'll have similar enough
features to the two's it's seen before, and it'll be able to recognize them. Turns out, yes, just a little bit
less accurately, around 80-95% depending on the run. If you press 1, it'll display the images it got incorrect,
and many of them are pretty reasonable: getting a 3 and an 8 mixed up, a 2 and a 7... Plus some of the test
images are really terribly drawn, to the point where I don't even know what digit it is
	Finally, a graphics window pops up for you to draw, like I've said before. It takes in your drawing, adds
some gray pixels around the ones you clicked to smooth it out and make it look more like the images it's seen
before, and then puts it through the network, telling you it's guess (the output neuron with the highest value)
and the confidence it has in that guess (the value of that neuron).

=============================================================================================================

CONCLUSION:
	Again, sorry for the long read. I didn't explain all the nitty gritty in the program, but that's conceptually
what happens to the best of my understanding. The 3Blue1Brown videos are great for explaining more on the math
end. As I mentioned, this is also not nearly the most efficient way to go about this problem. Most people use
matrix multiplication from linear algebra for all those neurons, but I wanted to get a grasp of what's actually 
happening one thing at a time, which is why I have all those beautiful FOR loops. Have fun playing around with it!
