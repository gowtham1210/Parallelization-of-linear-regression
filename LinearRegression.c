#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "csvparse.c"

double costFunction();
void meanNormalization();
void gradientDescent();

int main(int argc, char **argv)
{
	char *filename = argv[1];
	/*This is the number of features provided on the command line*/
	/*The csv should contain values for each x value and the result, per example*/
	int features = atoi(argv[2]);
	int examples = atoi(argv[3]); /*Number of training examples provided*/
	double cost = DBL_MAX;				/*Cost for the current hypothesis, set arbitratily high*/
	/*Values for the coefficients in the hypothesis function*/
	double *theta = malloc(features * sizeof(double));
	double **X = malloc(examples * sizeof(double *));
	for (int i = 0; i < examples; i++)
	{
		X[i] = malloc(features * sizeof(double));
	}
	double *Y = malloc(examples * sizeof(double));
	parse(features, examples, X, Y, filename);
	for (int i = 0; i < features; i++)
	{
		theta[i] = 0;
	}
	theta[0] = 1;
	double **meanAndRange = malloc((features - 1) * sizeof(double *));
	for (int i = 0; i < features - 1; i++)
	{
		meanAndRange[i] = malloc(2 * sizeof(double));
	}

	/*meanNormalization(X, Y, meanAndRange, features, examples);*/

	clock_t begin, end;
	begin = clock();
	double timeElapsed;
	gradientDescent(X, Y, theta, meanAndRange, features, examples);
	int *values = malloc((features - 1) * sizeof(int));
	values[0] = 1;
	char val[5];
	/*Print the learned formula*/
	printf("Learned function: %f", theta[0]);
	for (int i = 1; i < features; i++)
	{
		printf(" + %f(x%d)", theta[i], i);
	}
	printf("\n");
	end = clock();
	timeElapsed = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Elapsed Time: %f\n", timeElapsed);
	/*Obtain experimental values*/
	for (int i = 1; i < features; i++)
	{
		printf("Value for x%d:", i);
		scanf("%s", val);
		values[i] = atoi(val);
	}
	/*Print out the estimate for given values*/
	float output = 0;
	for (int i = 0; i < features; i++)
	{
		output += values[i] * theta[i];
	}
	printf("\nOutput: %f\n", output);
}

double costFunction(int *theta, double **X, double *Y, int features, int examples)
{
	double cost;
	double runningSum;
	for (int i = 0; i < examples; i++)
	{
		double xValue = 0;
		for (int j = 0; j < features; j++)
		{
			xValue += X[i][j] * theta[j];
		}
		runningSum += pow(xValue - Y[i], 2);
	}
	cost = (.5 * examples) * runningSum;
	return cost;
}

void meanNormalization(double **X, double **Y, double **meanAndRange, int features, int examples)
{
	double min;
	double max;
	double mean;
	for (int i = 1; i < features; i++)
	{
		min = X[0][i];
		max = X[0][i];
		mean = 0;
		for (int j = 0; j < examples; j++)
		{
			if (X[j][i] > max)
			{
				max = X[j][i];
			}
			if (X[j][i] < min)
			{
				min = X[j][i];
			}
			mean += X[j][i];
		}
		mean /= examples;
		meanAndRange[i - 1][0] = mean;
		meanAndRange[i - 1][1] = max - min;
	}
	for (int i = 0; i < examples; i++)
	{
		for (int j = 1; j < features; j++)
		{
			X[i][j] = (X[i][j] - meanAndRange[j - 1][0]) / meanAndRange[j - 1][1];
		}
	}
}
void gradientDescent(double **X, double *Y, double *theta, double **meanAndRange, int features, int examples)
{
	char iters[5];
	int iterations;
	double alpha = 0.001;
	double absCost;
	double hypothesis[examples];
	double runningSum;
	double gradients[examples];
	double intermediateCost;
	double previousCost = 0;
	printf("Gradient descent iterations(1-4 digits): ");
	scanf("%s", iters);
	iterations = atoi(iters);
	/*Iterates gradient descent iterations times*/
	for (int i = 0; i < iterations; i++)
	{
		/*initialize all the gradients to zero*/
		for (int i = 0; i < features; i++)
		{
			gradients[i] = 0;
		}
		/*Sets the values of the hypothesis, based on the current values of theta*/
		for (int a = 0; a < examples; a++)
		{
			runningSum = 0;
			for (int b = 0; b < features; b++)
			{
				runningSum += theta[b] * X[a][b];
			}
			hypothesis[a] = runningSum;
		}
		/*Actual gradient descent step- adjusts the values of theta by descending the gradient*/
		for (int j = 0; j < examples; j++)
		{
			intermediateCost = (hypothesis[j] - Y[j]);
			for (int l = 0; l < features; l++)
			{
				gradients[l] += intermediateCost * X[j][l];
			}
			for (int k = 0; k < features; k++)
			{
				theta[k] -= (alpha * gradients[k]) / examples;
			}
			absCost = fabs(intermediateCost);
			if (absCost > previousCost)
			{
				alpha /= 2;
			}
			else
			{
				alpha += .001;
			}
			previousCost = absCost;
		}
	}
}
