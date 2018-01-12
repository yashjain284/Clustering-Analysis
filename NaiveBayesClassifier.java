import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class NaiveBayesClassifier {
	
	public static HashMap<Integer, HashMap<Integer, String>> posteriorProbablities = new HashMap<Integer, HashMap<Integer, String>>(); //Map storing the Map of posterior probabilies for each label  
	public static HashMap<Integer, Double> classPrior = new HashMap<Integer, Double>(); //Map storing the class prior of each label.
	public static HashMap<String, Integer> descriptorPrior = new HashMap<String, Integer>(); //Map storing the descriptor prior for each feature value in the dataset.
	public static HashMap<Integer, Integer> truthMap = new HashMap<Integer, Integer>(); //Map storing the truth labels for data point(represented by line number).
	public static HashMap<Integer, Integer> resultMap = new HashMap<Integer, Integer>(); //Map storing the result labels for data point(represented by line number).
	public static HashMap<Integer, String[]> inputMap = new HashMap<Integer, String[]>(); //Map storing the data points for each line number.
	public static boolean[] isDiscrete; // boolean map to check if the type of the feture is discrete or not : True : Discrete or False : Continuous
	public static int kFactor = 10;
	public static int rowCount = 0;
	public static int featureCount = 0;
	
	//Generate Test ArrayList of Line Numbers selected in test Data in the Kth Iteration of K-Fold Validation.
	public static ArrayList<Integer> generateTestData(int k) {
		ArrayList<Integer> test = new ArrayList<Integer>();
		int start = (k - 1) * (rowCount / kFactor);
		int end = (k) * (rowCount / kFactor);
		for (; start < end; start++) {
			test.add(start);
		}
		return test;

	}
	
	//Generate Train ArrayList of Line Numbers selected in Train Data in the Kth Iteration of K-Fold Validation.
	public static ArrayList<Integer> generateTrainingData(int k) {
		ArrayList<Integer> train = new ArrayList<Integer>();
		if(k == -1) {
			for (int i = 0; i <= rowCount - 1; i++)
				train.add(i);
		}
		else {
			int start = (k - 1) * (rowCount / kFactor);
			int end = (k) * (rowCount / kFactor);
			for (int i = 0; i <= rowCount - 1; i++) {
				if (i >= start && i < end)
					continue;
				train.add(i);
			}
		}
		return train;
	}
	
	//Takes the input set and computes the posterior probability for each data point for each class and assigns the class to the higher probability.
	private static double[] testData(ArrayList<Integer> testSet) {
		// Testing data now.
		int a = 0; //To store the no of True Positives.
		int b = 0; //To store the no of False Postivies.
		int c = 0; //To store the no of False Negative.
		int d = 0; //To store the no of True Negative.
		for (int i : testSet) {
			String[] td = inputMap.get(i);
			double posterior0 = classPrior.get(0) * computePDF(td, 0); //Computing posterior for label 0
			double posterior1 = classPrior.get(1) * computePDF(td, 1); //Computing posterior for label 1
			int resultLabel = 0;
			if (posterior0 < posterior1)
				resultLabel = 1;
			resultMap.put(i, resultLabel); // Storing the result in the resultMap
			if (resultLabel == 1 && truthMap.get(i) == 1)
				a++;
			else if (resultLabel == 1 && truthMap.get(i) == 0)
				b++;
			else if (resultLabel == 0 && truthMap.get(i) == 1)
				c++;
			else
				d++;
		}
		//computing evaluation measure based on the result labels.
		double accuracy = ((double) (a + d) / (double) (a + b + c + d)) * 100;
		double precision = ((double) (a) / (double) (a + b)) * 100;
		double recall = ((double) (a) / (double) (a + c)) * 100;
		double fmeasure = ((double) (2 * a) / (double) (2 * a + b + c)) * 100;
		
		System.out.println("Accuracy : " + String.format("%.2f", accuracy) + "%");
		System.out.println("Precision : " + String.format("%.2f", precision) + "%");
		System.out.println("Recall : " + String.format("%.2f", recall) + "%");
		System.out.println("F-Measure : " + String.format("%.2f", fmeasure) + "%");
		
		double measures[] = { accuracy, precision, recall, fmeasure}; //Store it an an array to return it to the calling function.
		
		return measures;
	}

	/**Function to train the data. Given the data, it is split into as many parts as there are class labels. (Here it is 2)
	* We caompte the class Priors for the class Labels
	* We call the computePosterior for each split to train the features.
	* Store the posterior probailites for each split onto a Map of Maps(posteriorProbablities).
	*/
	private static void trainData(ArrayList<Integer> training) {

		ArrayList<String[]> tr1 = new ArrayList<String[]>();
		ArrayList<String[]> tr0 = new ArrayList<String[]>();
		for (int i = 0; i < training.size(); i++) {
			if (truthMap.get(training.get(i)) == 1)
				tr1.add(inputMap.get(training.get(i)));
			else
				tr0.add(inputMap.get(training.get(i)));
		}
		double denom = (double) tr1.size() + (double) tr0.size();
		classPrior.put(1, (double) tr1.size()/denom);
		classPrior.put(0, (double) tr0.size()/denom);

		String[][] trainDataLabel1 = tr1.toArray(new String[tr1.size()][featureCount]);
		String[][] trainDataLabel0 = tr0.toArray(new String[tr0.size()][featureCount]);
		
		HashMap<Integer, String> label0Map = computePosterior(trainDataLabel0);
		HashMap<Integer, String> label1Map = computePosterior(trainDataLabel1);

		posteriorProbablities.put(0, label0Map);
		posteriorProbablities.put(1, label1Map);
	}

	/**
	 * This function takes the test data point and computes the posterior probability 
	 * for Continuous Data : Computes the PDF formula using mean and variance precomputed in the map.
	 * for Catergorical Data : Computes using the posteriors stored in the map.
	 */
	private static double computePDF(String[] testSet, int label) {
		double posterior = 1.0;
		for (int i = 0; i < testSet.length; i++) {
			if (!isDiscrete[i]) {
				Double val = Double.parseDouble(testSet[i]);
				String str = posteriorProbablities.get(label).get(i);
				String sp[] = str.split(",");
				double mean = Double.parseDouble(sp[0]);
				double variance = Double.parseDouble(sp[1]);
				double expValue = (-1 * Math.pow(val - mean, 2)) / (2 * variance);
				double pdf = Math.exp(expValue) / Math.sqrt(2 * Math.PI * variance);
				posterior = posterior * pdf;
			} 
			else {
				String discreteVal = testSet[i];
				String str = posteriorProbablities.get(label).get(i);
				Double prob = 0.0;
				String[] sp = str.split(",");
				for (String s : sp) {
					String[] split = s.split("_");
					if (split[0].equals(discreteVal))
						prob = Double.parseDouble(split[1]);
				}
				posterior = posterior * prob;
			}
		}
		return posterior;
	}

	/**
	 * This function trainData split as the input and trains the probablistic model using storing the data in the posteriorMap
	 * Data Matrix is transposed so that we work column wise.
	 * for Continuous Data : Computes the mean and variance for the festure column.
	 * for Catergorical Data : Computes using the posteriors probabilities for each value of each feature and store in the posteriorMap.
	 */
	public static HashMap<Integer, String> computePosterior(String[][] trainData) {
		HashMap<Integer, String> posteriorMap = new HashMap<Integer, String>();
		String[][] dataTranspose = StatisticalHelper.transposeMatrix(trainData);
		for (int i = 0; i < featureCount; i++) {
			if (isDiscrete[i]) {
				String str = generateDiscretePosteriors(dataTranspose[i]);
				posteriorMap.put(i, str);
				continue;
			}
			double mean = StatisticalHelper.computeMean(dataTranspose[i]);
			double variance = StatisticalHelper.computeVariance(dataTranspose[i], mean);
			posteriorMap.put(i, mean + "," + variance);
		}
		return posteriorMap;
	}

	/**
	 * This function takes each feature and computes the Posterior for Discrete data computing P(X=x1|C) during testing.
	 * Computes using the posteriors probabilities for each value of each feature and store in the posteriorMap.
	 */
	private static String generateDiscretePosteriors(String[] feature) {
		HashMap<String, Integer> valueMap = new HashMap<String, Integer>(); //Storing the frequency of each value of the feature
		for (int i = 0; i < feature.length; i++) {
			if (valueMap.containsKey(feature[i]))
				valueMap.put(feature[i], valueMap.get(feature[i]) + 1);
			else
				valueMap.put(feature[i], 1);
		}
		String str = "";
		//Iterate through the unique keys (or feature values) to find the probability.
		for (String key : valueMap.keySet()) {
			double prob = (double) valueMap.get(key) / (double) feature.length; //Computing the descriptor probability for each value.
			String entry = key + "_" + prob;
			str = str + entry + ",";
		}
		return str.substring(0, str.length() - 1); //Returning it as a string to the caller function.
	}

	/**
	 * This function takes each feature and computes the Decriptor prior for computing P(X) during testing.
	 * Computes using the probabilities for each value of each feature and store in the posteriorMap.
	 */
	private static void trainDescriptorPrior(ArrayList<Integer> training) {
		
		ArrayList<String[]> tr = new ArrayList<String[]>();
		for (int i = 0; i < training.size(); i++) {//Populating the training set.
				tr.add(inputMap.get(training.get(i)));
		}
		String[][] trainData = tr.toArray(new String[tr.size()][featureCount]); //Comverting to array
		String[][] trainTranspose = StatisticalHelper.transposeMatrix(trainData); //Computing transpose to work with feature columns
		for(int i=0;i<trainTranspose.length;i++) { //Iterating through each feature row.
			String[] feature = trainTranspose[i];  
			for (int j = 0; j < feature.length; j++) { //For each feature we populate our descriptorPrior map with the frequency of each value.
				String key = feature[j];  
				if (descriptorPrior.containsKey(key))
					descriptorPrior.put(key, descriptorPrior.get(key) + 1);
				else
					descriptorPrior.put(key, 1);
			}
		}
	}
	

	/**
	 * This function takes each the test data point and computes the descriptor P(X)
	 * Computes using the probabilities for each value of each feature and store in the posteriorMap.
	 */
	private static double computeDescriptorPrior(String[] values) {
		double prior = 1.0;
		for(int i=0;i<values.length;i++) {
			double prob = descriptorPrior.get(values[i])/(double)rowCount; //Calls the descriptor map. and computed the descriptor probability.
			prior = prior*prob;
		}
		return prior;
	}
	
	//Function which takes the 1st row of data and checks the type for each features and stores it in the boolean map.
	private static int determineFeatureTypes(String[] data) {

		isDiscrete = new boolean[data.length];
		for (int i = 0; i < data.length; i++) {

			try {
				double val = Double.parseDouble(data[i]);
				isDiscrete[i] = false;
			} catch (Exception e) {
				isDiscrete[i] = true;
			}
		}
		return data.length;
	}
	
	//Function to start k Fold vailation. K=10/
	public static void kFoldCrossValidation() {
		
		double acc = 0.0, pre = 0.0, recall = 0.0, fmeasure = 0.0;
		for (int i = 1; i <= 10; i++) {

			System.out.println("\n\n*********************************");
			System.out.println("FOR K = " + i);
			System.out.println("*********************************");
			ArrayList<Integer> testSet = generateTestData(i); //for each K we generate test data
			ArrayList<Integer> trainingSet = generateTrainingData(i);  //for each K we generate training data
			trainData(trainingSet); //Call the trainData passing the trainData set for the current value of K.
			double measures[] = testData(testSet); //We get the array if all the 4 evaluation measure as the result.
			
			//Update the running total.
			acc += measures[0];
			pre += measures[1];
			recall += measures[2];
			fmeasure += measures[3];
		}
		//Computing average measure for all the iteration of ks
		double avgAcc = acc / 10.0;
		double avgPre = pre / 10.0;
		double avgRecall = recall / 10.0;
		double avgFmeasure = fmeasure / 10.0;
		System.out.println("\n\n*********************************");
		System.out.println("AVERAGE MEASURES");
		System.out.println("*********************************");
		System.out.println("Average Accuracy : " + String.format("%.2f", avgAcc) + "%");
		System.out.println("Average Precision : " + String.format("%.2f", avgPre) + "%");
		System.out.println("Average Recall : " + String.format("%.2f", avgRecall) + "%");
		System.out.println("Average F-Measure : " + String.format("%.2f", avgFmeasure) + "%");
		
	}
	
	//Function to read the query during Project Demo
	public static double[] readQuery(String str) {
		
		double result[] = new double[2];
		String discreteValues[] = str.split(",");
		//Computing the posterior probaility and mulitplying with the class prior.
		result[0] = classPrior.get(0)*computePDF(discreteValues,0); 
		result[1] = classPrior.get(1)*computePDF(discreteValues,1);
		double descriptor = computeDescriptorPrior(str.split(",")); //Calls the computeDescriptorPrior and passes the data point as a array of features.
		//Diving the result by the computed descriptor P(X)
		result[0] /= descriptor; 
		result[1] /= descriptor;
		return result;
	}

	/**Function to read the input dataset file passed in the argment.
	 * Populate the truthMap with the line number to true data labels.
	 * Populate inputMap with the line number to data point
	 */
	public static void readFile(String fileURL) {
		try {

			BufferedReader br = new BufferedReader(new FileReader(fileURL));
			String line = "";
			int lineNo = 0;
			while ((line = br.readLine()) != null) {
				int len = line.length();
				String[] data = line.substring(0, len - 2).split("\t");
				int label = Integer.parseInt(line.substring(line.lastIndexOf("\t") + 1, len));
				truthMap.put(lineNo, label);//populate the truthMap.
				inputMap.put(lineNo, data);//populate the inputMap.
				lineNo++;
			}
			rowCount = lineNo;
			featureCount = determineFeatureTypes(inputMap.get(1)); //Call the determineFeatureTypes to initialize the boolean map.
			br.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		
		String fileName= "";
		Scanner sc = new Scanner(System.in);
		System.out.println("Please select Database..\n 1: Data Set 1 \n 2: Data Set 2 \n 4: Data Set 4");
		System.out.print("Please Enter your choice : ");
		int ch = Integer.parseInt(sc.nextLine());
		if(ch == 1)
			fileName = "project3_dataset1.txt";
		else if(ch == 2)
			fileName = "project3_dataset2.txt";
		else if(ch == 4)
			fileName = "project3_dataset4.txt";
		
		long time1 = System.currentTimeMillis();
		readFile(fileName);
		if(!fileName.equals("project3_dataset4.txt"))
			kFoldCrossValidation();
		else {
			ArrayList<Integer> trainingSet = generateTrainingData(-1);
			trainData(trainingSet);
			trainDescriptorPrior(trainingSet);
			System.out.print("Please enter the query you want to classify :");
			String str = sc.nextLine();
			time1 = System.currentTimeMillis();
			double result[] = readQuery(str);	
			System.out.println("P(\""+str+"\"/0) : "+String.format("%.4f", result[0]));
			System.out.println("P(\""+str+"\"/1) : "+String.format("%.4f", result[1]));
			String label;
			if(result[0] > result[1])
				label  = "0";
			else
				label  = "1";
			System.out.println("The data point is give the class label : "+label);
		}
		long time2 = System.currentTimeMillis();
		long timeTaken = time2 - time1;  
		System.out.println("Time taken " + timeTaken + " ms");
		sc.close();
	}
}