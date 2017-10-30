#include<stdio.h>
#include<math.h>
#include<omp.h>

#define NUM_THREADS 8

const TOTAL_SET_SIZE = 209;
const TRAINING_DATA_SIZE = 180;
const FEATURE_SIZE = 7;


    static double data[209][7]; // Array for training data
	static double alpha = 0.001; // Learning Rate
	static int iterations=0;
	static double w[] = {1,1,1,1,1,1,1};
	static int w_length = 7;

	//Function for reading file and entering the training data into the array
   void readFile()
	{
        int a=0,b=0;
        double c;
	    FILE *file;
        file=fopen("machine.txt","r");
        if(file!=NULL){
        do{
        if(a<6){
        data[b][a]=c;a++;

        }
        else{
        data[b][a]=c;a=0;
        b++;
        }
        }while(fscanf(file,"%lf",&c)!=EOF);


	}
	else{
        printf("File error!!!");
	}

}

	//Method for finding the predicted value based on theobjective function
    double expr(int k)
	{
		double result = 0;
		int i = 0;
	
				result = result + w[i];
		#pragma omp for schedule(static,chunk) reduction (+:result)
		for(i=1;i<w_length;i++)
		{
			
			
				result = result + w[i] * data[k][i-1]; // Calculating the value of the expression by substituting the data of kth row in the objective function equation
			

		}
        printf("Result: %lf\n",result);
		return result;
	}

	//Method for calculating loss function of a given feature's coefficient in the objective function eqn
    double lossfunction(int index)
	{
        int i,k = 0;
		double lossfunctionvalue[FEATURE_SIZE];
		
		lossfunctionvalue[k] = lossfunctionvalue[k] + (-1)*(data[i][(FEATURE_SIZE-1)]-expr(i)); // for w0, the differentiation gives (-1)
				
		#pragma omp for schedule(static,chunk) reduction (+:lossfunctionvalue)
		for(i=0;i<TRAINING_DATA_SIZE;i++)
		{
			for(k=1;k<FEATURE_SIZE;k++)
			{
				
				lossfunctionvalue[k] = lossfunctionvalue[k] + (-1)*data[i][k-1]*(data[i][(FEATURE_SIZE-1)]-expr(i)); // for other w coefficients
	
			}

		}
	return lossfunctionvalue[index]; // This function returns an array of loss functions for each feature's coefficient

	}

	// This method is used for finding the max of a feature in the training data
	// This max value would be used in the feature scaling of the training data
	 double findmax(double max[])
	{

		int i,j;
		int lmax = 0;
		for(i=0;i<(FEATURE_SIZE-1);i++)
		{
			max[i] = 0;
		}

		 
		for(j=0;j<(FEATURE_SIZE-1);j++)
		{
		#pragma omp for schedule(static,chunk)
			for(i=0;i<TOTAL_SET_SIZE;i++)
			{


				if(data[i][j] > lmax)
				{

					lmax = data[i][j];

				}

			}
			
			if(max[j] < lmax)
			{
			#pragma omp critical
			max[j] = lmax;
			}
		}


	}



//     There are three ways of feature scaling for any training data set I have tested for two methods among them,
//	one by finding max value by findmax() method and the other by finding mean and std deviation by the following methods.
//	Feature scaling by finding max value is yielding results way too faster than by mean and std deviation and the results I got
//	on the data set were similar. So I have commented this implementation.




//	 double mean()
//	{
//		double meanvalue[]=new double[(FEATURE_SIZE-1)];
//		for(int i=0;i<(FEATURE_SIZE-1);i++)
//		{
//			meanvalue[i] = 0;
//		}
//		for(int j=0;j<(FEATURE_SIZE-1);j++)
//		{
//			for(int i=0;i<TOTAL_SET_SIZE;i++)
//			{
//				meanvalue[j] = (meanvalue[j] + data[i][j])/TOTAL_SET_SIZE;
//			}
//		}
//
//		return meanvalue;
//	}
//
//	 double stddev()
//	{
//		double sd[] = new double[(FEATURE_SIZE-1)];
//		for(int i=0;i<(FEATURE_SIZE-1);i++)
//		{
//			sd[i] = 0;
//		}
//		double meanvalues[] = mean();
//		for(int j=0;j<(FEATURE_SIZE-1);j++)
//		{
//			for(int i=0;i<TOTAL_SET_SIZE;i++)
//			{
//				sd[j] = sd[j]+ (data[i][j] - meanvalues[j]);
//			}
//		}
//
//		return sd;
//	}

	//Method for feature scaling
	 void featureScaling()
	{
	    double max[(FEATURE_SIZE-1)];
	    int i,j;
    	findmax(&max); //Max array is retrieved by the findmax() method
//		double meanvalues[] = mean();
//		double sd[] = stddev(); // Implementation for the second method i.e by mean and std deviation
		#pragma omp for schedule(static,chunk) reduction(/:data)
		for(i=0;i<(FEATURE_SIZE-1);i++)
		{

			for(j=0;j<TOTAL_SET_SIZE;j++)
			{
				data[j][i] = (data[j][i])/max[i]; // All the dataset's features are divided by their corresponding max values making all the values between 0 and 1

//				data[j][i] = (data[j][i] - meanvalues[i])/sd[i];  // Implementation for the second method i.e by mean and std deviation
			}
		}
	}


	// Update rule for the coefficients

	double abs(double num)
	{

	    return num > 0 ? num : (-1)*num;
	}
	 void update()
	{
		// An array to store the previous values of the coefficients so that we can compare them and stop the loop when they are equal
		double prev[FEATURE_SIZE];

		// Comparision variable so the absolute difference between the previous and the current values has to be 0 for the loop to end.
		// Although the difference isn't getting exactly to 0, It is considered as 0 as soon as the difference reaches a value of the order exp(-15)
		double mindiff = 0;
        int i;
    printf("prev: %lf",abs(prev[0]-w[0]));
		while( (abs( (prev[0]-w[0]) ) > mindiff) || (abs( (prev[1]-w[1]) ) > mindiff)
				|| ( abs( (prev[2]-w[2]) ) > mindiff) || ( abs( (prev[3]-w[3]) ) > mindiff)
				|| ( abs( (prev[4]-w[4]) ) > mindiff) || ( abs( (prev[5]-w[5]) ) > mindiff)
				|| ( abs( (prev[6]-w[6]) ) > mindiff) ) // Condition check for the previous and current values of the coefficients
		{
		#pragma omp for schedule(static,chunk) reduction(-:w)
			for( i=0;i<w_length;i++)
			{
				prev[i] = w[i]; // Assigning the old value to the prev array

				w[i] = w[i] - alpha * lossfunction(i); // Applying the update rule by multiplying the learning rate alpha and calling the lossfunction on every iteration
				printf("Loss Function value for w%d: %lf\n",i,lossfunction(i));  // Printing the loss function values
				iterations++; // Calculating no of iterations taken to finally get the solution
			}

			for( i=0;i<FEATURE_SIZE;i++)
			{
				printf("Difference: %lf\n",abs( (prev[i]-w[i]))); // Printing the difference between the previous and current values of the coefficients
			}

		}


	}

	 void test()
	{
	    int i;

		for( i=TRAINING_DATA_SIZE;i<TOTAL_SET_SIZE;i++)
		{
			printf("Original Result: %lf \t\tPredicted Value: %lf\n",data[i][(FEATURE_SIZE-1)],abs(expr(i)));
		}

	}
 main() {
		// TODO Auto-generated method stub


        int i,j;
		
		readFile(); // Passing the url of the data file for copying the data into the array
		omp_set_num_threads(NUM_THREADS);
		#pragma omp parallel default(shared) 
		{
		featureScaling(); // Calling the method for feature scaling of data

		update(); // Calling the update method to apply the update rule
		}
		printf("Final values of the objective function: ");

		for(i=0;i<FEATURE_SIZE;i++)
		{
			printf("w%d: %lf",i,w[i]); // Printing the final values of the coeffiecients
		}

		printf("No of iterations: %d",iterations); // Printing no of iterations

	//	test();



	}

