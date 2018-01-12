import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

class RFNode {
	int col, label = -1;
	String value = "";
	boolean isLeaf = false;
	RFNode left = null, right= null;
	
	public RFNode() { }
	
	public RFNode setCol(int x) {
		col = x;
		return this;
	}
	
	public RFNode setValue(String x) {
		value = x;
		return this;
	}
	
	public RFNode setLabel(int x) {
		label = x;
		isLeaf = true;
		return this;
	}
}

public class RandomForest {
	
	public static HashMap<Integer, Integer> actualClusters = new HashMap<>();
	public static HashMap<Integer, Integer> resultClusters = new HashMap<>();
	public static HashMap<Integer, ArrayList<String>> originalData = new HashMap<>();
	public static HashMap<Integer, Integer> columnDataType = new HashMap<>();
	public static int attributes = 0;
	public static int dataSize = 0;
	public static int noOfTrees;
	public static int noOfAttr;
	public static ArrayList<Integer> currAttributes = new ArrayList<>();
	public static HashMap<Integer, HashMap<Integer, Integer>> currResultClusters = new HashMap<>();
	public static Random randomGenerator = new Random();
	public static ArrayList<Double> accuracy = new ArrayList<>();
	public static ArrayList<Double> precision = new ArrayList<>();
	public static ArrayList<Double> recall = new ArrayList<>();
	public static ArrayList<Double> fmeasure = new ArrayList<>();
	public static int folds = 10;
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		System.out.println("Enter file name");
		String file = sc.nextLine();
		System.out.println("Enter no of folds");
		folds = Integer.parseInt(sc.nextLine());
		System.out.println("Enter no of trees");
		noOfTrees = Integer.parseInt(sc.nextLine());
		System.out.println("Enter no of attributes");
		noOfAttr = Integer.parseInt(sc.nextLine());
		fileRead(file);
		RandomForestAlgo();
		calcAverage();
	}

	/**
	 * It reads the files, and stores the important information like data and 
	 * true label in arraylist and hashmap.
	 * @param file filename
	 */
	private static void fileRead(String file) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line = "";
			boolean flag = true;
			int count = 0;
			while((line=br.readLine())!=null) {
				count++;
				String data[] = line.split("\\t");
				ArrayList<String> temp = new ArrayList<>();
				int i=0;
				for(i=0;i<data.length-1;i++) {
					temp.add(data[i]);
					if(flag) {
						try {
							Double.parseDouble(data[i]);
							columnDataType.put(i, 0);
						} catch(Exception e) {
							columnDataType.put(i, 1);
						}
					}
				}
				flag = false;
				attributes = data.length-1;
				originalData.put(count, temp);
				int label =  Integer.parseInt(data[i]);
				actualClusters.put(count, label);
			}
			dataSize = actualClusters.size();
			br.close();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * for each fold, it builds the tree, tests the test data and calculates accuracy,
	 * precision, recall and fmeasure. 
	 */
	private static void RandomForestAlgo() {
		
		int i, dataPerIteration = dataSize/10;
		int testStartRow = 0, testEndRow =0;
		for(i=1;i<=folds;i++) {
			System.out.println("Fold = "+i);
			testStartRow = testEndRow+1;
			testEndRow += dataPerIteration;
			if(i == folds)
				testEndRow = dataSize;
			ArrayList<Integer> preTrainData = new ArrayList<>(originalData.keySet());
			ArrayList<Integer> testData = new ArrayList<>();
			for(int k = testStartRow;k<=testEndRow;k++) {
				preTrainData.remove(testStartRow-1);
				testData.add(k);
			}
			currResultClusters = new HashMap<>();
			for(int x=0;x<noOfTrees;x++) {
				ArrayList<Integer> trainData = new ArrayList<>();
				for(int k=0;k<preTrainData.size();k++) {
					int index = randomGenerator.nextInt(preTrainData.size());
					int row = preTrainData.get(index);
					if(!trainData.contains(row))
						trainData.add(row);
				}
				RFNode root = null;
				root = buildTree(trainData);
				testData(root, testData);
			}
			finalTestData();
			getAccuracy(testData);
			getPrecision(testData);
			getRecall(testData);
			getFMeasure(testData);
		}
	}
	
	private static void finalTestData() {
		for (Map.Entry<Integer, HashMap<Integer, Integer>> entry : currResultClusters.entrySet()) {
			int row = entry.getKey();
			int zeroClass = 0, oneClass = 0;
			if(entry.getValue().containsKey(0))
				zeroClass = entry.getValue().get(0);
			if(entry.getValue().containsKey(1))
				oneClass = entry.getValue().get(1);
			if(zeroClass > oneClass)
				resultClusters.put(row, 0);
			else
				resultClusters.put(row, 1);
		}
	}
	
	/**
	 * For each test data, it finds the label using tree and stores in a map
	 * @param root Decision tree 
	 * @param testData Collection of test data
	 */
	private static void testData(RFNode root, ArrayList<Integer> testData) {
		for(int i=0;i<testData.size();i++) {
			int testRow = testData.get(i);
			int outputLabel = getLabel(root, originalData.get(testRow));
			HashMap<Integer, Integer> temp = new HashMap<>();
			if(currResultClusters.containsKey(testRow))
				temp = currResultClusters.get(testRow);
			int count = 0;
			if(temp.containsKey(outputLabel))
				count = temp.get(outputLabel);
			count++;
			temp.put(outputLabel, count);
			currResultClusters.put(testRow, temp);
		}
	}
	
	/**
	 * Traverses the tree along the path based on condition
	 * @param root Decision tree
	 * @param data: a test data 
	 * @return predicted label
	 */
	private static int getLabel(RFNode root, ArrayList<String> data) {
		if(root.isLeaf) {
			return root.label;
		}
		int datatype = columnDataType.get(root.col);
		String val = data.get(root.col);
		if(datatype == 0) {
			if(Double.parseDouble(val) <= Double.parseDouble(root.value))
				return getLabel(root.left, data);
			else
				return getLabel(root.right, data);
		} else {
			if(val.equals(root.value))
				return getLabel(root.left, data);
			else
				return getLabel(root.right, data);
		}
	}
	
	/**
	 * It finds best split, partitions the data as per split and recursively builds tree.
	 * @param trainData Collection of training data
	 * @return Decision tree root
	 */
	private static RFNode buildTree(ArrayList<Integer> trainData) {
		String output[] = findBestSplit(trainData).split(";");
		double bestGain = Double.parseDouble(output[0]);
		int bestGainRow = Integer.parseInt(output[1]), bestGainCol = Integer.parseInt(output[2]);
		String bestGainVal = "";
		try {
			bestGainVal = output[3];
		} catch(Exception e) {
			e.printStackTrace();
		}
		if(bestGain == 0) {
			RFNode result = new RFNode();
			int label = actualClusters.get(trainData.get(0));
			result.setCol(bestGainCol).setValue(bestGainVal).setLabel(label);
			return result;
		}
		RFNode root = new RFNode();
		root.setCol(bestGainCol).setValue(bestGainVal);
		ArrayList<Integer> left = new ArrayList<>();
		ArrayList<Integer> right = new ArrayList<>();
		partition(trainData,bestGainCol,bestGainVal,left,right);
		root.left = buildTree(left);
		root.right = buildTree(right);
		return root;
	}
	
	/**
	 * For each value, it find the information gain and returns the value which gives best gain
	 * @param trainData Collection of training data
	 * @return best gain with corresponding column and value
	 */
	private static String findBestSplit(ArrayList<Integer> trainData) {
		double bestGain = Double.NEGATIVE_INFINITY;
		int bestGainRow = -1, bestGainCol = -1;
		String bestGainVal = "";
		double currUncertainty = calcGini(trainData);
		ArrayList <Integer>currAttributes2 = new ArrayList<>();
		for(int y=0;y<noOfAttr;y++) {
			int attr = randomGenerator.nextInt(attributes);
			if(currAttributes2.contains(attr))
				y--;
			else
				currAttributes2.add(attr);
		}
		for(int i=0;i<trainData.size();i++) {
			for(int x=0;x<currAttributes2.size();x++) {
				int col = currAttributes2.get(x);
				int row = trainData.get(i);
				String value = originalData.get(row).get(col);
				ArrayList<Integer> left = new ArrayList<>();
				ArrayList<Integer> right = new ArrayList<>();
				partition(trainData,col,value,left,right);
				double currGain = calcGain(left, right, currUncertainty);
				if(currGain >= bestGain) {
					bestGain = currGain;
					bestGainRow = row;
					bestGainCol = col;
					bestGainVal = value;
				}
			}
		}
		return (bestGain+";"+bestGainRow+";"+bestGainCol+";"+bestGainVal);
	}
	
	/**
	 * partitions data into left and right based on column and its attributes.
	 */
	private static void partition(ArrayList<Integer> trainData, int column, String value, ArrayList<Integer> left, ArrayList<Integer> right) {
		int datatype = columnDataType.get(column);
		for(int i=0;i<trainData.size();i++) {
			int row = trainData.get(i);
			String val = originalData.get(row).get(column);
			if(datatype == 0) {
				if(Double.parseDouble(val) <= Double.parseDouble(value))
					left.add(row);
				else
					right.add(row);
			} else if(datatype == 1) {
				if(val.equals(value))
					left.add(row);
				else
					right.add(row);
			}
		}
	}
	
	/**
	 * Calculates gini at a node
	 * @param rows Data at node
	 * @return gini index
	 */
	private static double calcGini(ArrayList<Integer> rows) {
		HashMap<Integer, Integer> labelCount = new HashMap<>();
		for(int i=0;i<rows.size();i++) {
			int row = rows.get(i);
			int currlabel = actualClusters.get(row);
			if(labelCount.containsKey(currlabel))
				labelCount.put(currlabel, labelCount.get(currlabel)+1);
			else
				labelCount.put(currlabel,1);
		}
		double result = 1;
		for (Map.Entry<Integer, Integer> entry : labelCount.entrySet()) {
			double temp = (entry.getValue()*1.0)/rows.size();
			result -= (temp*temp);
		}
		return result;
	}
	
	/**
	 * Calculates the information gain at a node.
	 * @param left Left subtree data
	 * @param right right subtree data
	 * @param currUncertainty parent gain
	 * @return information gain
	 */
	private static double calcGain(ArrayList<Integer> left, ArrayList<Integer> right, double currUncertainty) {
		double leftGini = calcGini(left);
		double rightGini = calcGini(right);
		double p = (left.size()*1.0)/(left.size()+right.size());
		return currUncertainty - (p * leftGini) - ((1-p) * rightGini);
	}
	
	/**
	 * Calculates the accuracy for test data.
	 */
	private static void getAccuracy(ArrayList<Integer> testData) {
		int count = 0;
		for(int k=0;k<testData.size();k++) {
			int i = testData.get(k);
			if(resultClusters.get(i) == actualClusters.get(i))
				count++;
		}
		double acc = (count * 100.0)/testData.size();
		System.out.println("Accuracy = "+acc);
		accuracy.add(acc);
	}
	
	/**
	 * Calculates the Precision for test data.
	 */
	private static void getPrecision(ArrayList<Integer> testData) {
		int countA = 0, countC = 0;
		for(int k=0;k<testData.size();k++) {
			int i = testData.get(k);
			if(resultClusters.get(i) == actualClusters.get(i) && actualClusters.get(i) == 1)
				countA++;
			else if (resultClusters.get(i) != actualClusters.get(i) && actualClusters.get(i) == 0)
				countC++;
		}
		double Prec = (countA * 100.0)/(countA + countC);
		System.out.println("Precision = "+Prec);
		precision.add(Prec);
	}
	
	/**
	 * Calculates the Recall for test data.
	 */
	private static void getRecall(ArrayList<Integer> testData) {
		int countA = 0, countB = 0;
		for(int k=0;k<testData.size();k++) {
			int i = testData.get(k);
			if(resultClusters.get(i) == actualClusters.get(i) && actualClusters.get(i) == 1)
				countA++;
			else if (resultClusters.get(i) != actualClusters.get(i) && actualClusters.get(i) == 1)
				countB++;
		}
		double Rec = (countA * 100.0)/(countA + countB);
		System.out.println("Recall = "+Rec);
		recall.add(Rec);
	}
	
	/**
	 * Calculates the Fmeausre for test data.
	 */
	private static void getFMeasure(ArrayList<Integer> testData) {
		int countA = 0, countB = 0, countC = 0;
		for(int k=0;k<testData.size();k++) {
			int i = testData.get(k);
			if(resultClusters.get(i) == actualClusters.get(i) && actualClusters.get(i) == 1)
				countA++;
			else if (resultClusters.get(i) != actualClusters.get(i) && actualClusters.get(i) == 1)
				countB++;
			else if (resultClusters.get(i) != actualClusters.get(i) && actualClusters.get(i) == 0)
				countC++;
		}
		double FMeasure = (countA * 200.0)/(2*countA + countB + countC);
		System.out.println("FMeasure = "+FMeasure);
		fmeasure.add(FMeasure);
	}
	
	/**
	 * Calculates the average accuracy, precision, recall and fmeasure for entire data set
	 */
	private static void calcAverage() {
		double acc=0, prec =0,rec = 0, fmea = 0;
		for(int i=0;i<folds;i++) {
			acc += accuracy.get(i);
			prec += precision.get(i);
			rec += recall.get(i);
			fmea += fmeasure.get(i);
		}
		System.out.println("Average accuracy = "+acc/folds);
		System.out.println("Average precision = "+prec/folds);
		System.out.println("Average recall = "+rec/folds);
		System.out.println("Average fmeasure = "+fmea/folds);
	}
	
}
