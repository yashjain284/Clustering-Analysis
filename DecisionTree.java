import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Scanner;

/**
 * @author akshay
 * Node class to store the label, column and its values.
 */
class Node {
	int col, label = -1;
	String value = "";
	boolean isLeaf = false;
	Node left = null, right= null;
	
	public Node() { }
	
	public Node setCol(int x) {
		col = x;
		return this;
	}
	
	public Node setValue(String x) {
		value = x;
		return this;
	}
	
	public Node setLabel(int x) {
		label = x;
		isLeaf = true;
		return this;
	}
}

public class DecisionTree {
	
	public static HashMap<Integer, Integer> actualClusters = new HashMap<>();
	public static HashMap<Integer, Integer> resultClusters = new HashMap<>();
	public static HashMap<Integer, ArrayList<String>> originalData = new HashMap<>();
	public static HashMap<Integer, Integer> columnDataType = new HashMap<>();
	public static int attributes = 0;
	public static int dataSize = 0;
	public static int folds = 10;
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		System.out.println("Enter file name");
		String file = sc.nextLine();//"project3_dataset1.txt";\
		System.out.println("Enter no of folds");
		folds = Integer.parseInt(sc.nextLine());
		fileRead(file);
		CART();
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
	private static void CART() {
		int i, dataPerIteration = dataSize/10;
		int testStartRow = 0, testEndRow =0;
		for(i=1;i<=folds;i++) {
			testStartRow = testEndRow+1;
			testEndRow += dataPerIteration;
			if(i==folds)
				testEndRow = dataSize;
			ArrayList<Integer> trainData = new ArrayList<>(originalData.keySet());
			ArrayList<Integer> testData = new ArrayList<>();
			for(int k = testStartRow;k<=testEndRow;k++) {
				trainData.remove(testStartRow-1);
				testData.add(k);
			}
			Node root = null;
			root = buildTree(trainData);
			testData(root, testData);
//			levelOrder(root);
		}
		getAccuracy();
		getPrecision();
		getRecall();
		getFMeasure();
	}
	
	/**
	 * For each test data, it finds the label using tree and stores in a map
	 * @param root Decision tree 
	 * @param testData Collection of test data
	 */
	private static void testData(Node root, ArrayList<Integer> testData) {
		for(int i=0;i<testData.size();i++) {
			int testRow = testData.get(i);
			int outputLabel = getLabel(root, originalData.get(testRow));
			resultClusters.put(testRow, outputLabel);
		}
	}

	/**
	 * Traverses the tree along the path based on condition
	 * @param root Decision tree
	 * @param data: a test data 
	 * @return predicted label
	 */
	private static int getLabel(Node root, ArrayList<String> data) {
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
	private static Node buildTree(ArrayList<Integer> trainData) {
		String output[] = findBestSplit(trainData).split(";");
		double bestGain = Double.parseDouble(output[0]);
		int bestGainRow = Integer.parseInt(output[1]), bestGainCol = Integer.parseInt(output[2]);
		String bestGainVal = output[3];
		if(bestGain == 0) {
			Node result = new Node();
			int label = actualClusters.get(trainData.get(0));
			result.setCol(bestGainCol).setValue(bestGainVal).setLabel(label);
			return result;
		}
		Node root = new Node();
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
		double bestGain = 0;
		int bestGainRow = -1, bestGainCol = -1;
		String bestGainVal = "";
		double currUncertainty = calcGini(trainData);
		for(int i=0;i<trainData.size();i++) {
			for(int col=0;col<attributes;col++) {
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
	 * Calculates the accuracy for test data.
	 */
	private static void getAccuracy() {
		int count = 0;
		for(int i=1;i<=dataSize;i++) {
			if(resultClusters.get(i) == actualClusters.get(i))
				count++;
		}
		double accuracy = (count * 100.0)/dataSize;
		System.out.println("Accuracy = "+accuracy);
	}
	
	/**
	 * Calculates the precision for test data.
	 */
	private static void getPrecision() {
		int countA = 0, countC = 0;
		for(int i=1;i<=dataSize;i++) {
			if(resultClusters.get(i) == actualClusters.get(i) && actualClusters.get(i) == 1)
				countA++;
			else if (resultClusters.get(i) != actualClusters.get(i) && actualClusters.get(i) == 0)
				countC++;
		}
		double Precision = (countA * 100.0)/(countA + countC);
		System.out.println("Precision = "+Precision);
	}
	
	/**
	 * Calculates the recall for test data.
	 */
	private static void getRecall() {
		int countA = 0, countB = 0;
		for(int i=1;i<=dataSize;i++) {
			if(resultClusters.get(i) == actualClusters.get(i) && actualClusters.get(i) == 1)
				countA++;
			else if (resultClusters.get(i) != actualClusters.get(i) && actualClusters.get(i) == 1)
				countB++;
		}
		double Recall = (countA * 100.0)/(countA + countB);
		System.out.println("Recall = "+Recall);
	}
	
	/**
	 * Calculates the fmeasure for test data.
	 */
	private static void getFMeasure() {
		int countA = 0, countB = 0, countC = 0;
		for(int i=1;i<=dataSize;i++) {
			if(resultClusters.get(i) == actualClusters.get(i) && actualClusters.get(i) == 1)
				countA++;
			else if (resultClusters.get(i) != actualClusters.get(i) && actualClusters.get(i) == 1)
				countB++;
			else if (resultClusters.get(i) != actualClusters.get(i) && actualClusters.get(i) == 0)
				countC++;
		}
		double FMeasure = (countA * 200.0)/(2*countA + countB + countC);
		System.out.println("FMeasure = "+FMeasure);
	}
	
	/**
	 * to print the tree using level order traversal
	 * @param root
	 */
	public static void levelOrder(Node root) {
		Queue<Node> q = new LinkedList<Node>();
		q.add(root);
		while (true) {
			int nodeCount = q.size();
			if (nodeCount == 0)
				break;
			while (nodeCount > 0) {
				Node node = q.peek();
				if(node.isLeaf)
					System.out.print("leaf with label = "+node.label+";  ");
				else
					System.out.print("col = "+node.col + " value = "+node.value+";  ");
				q.remove();
				if (node.left != null)
					q.add(node.left);
				if (node.right != null)
					q.add(node.right);
				nodeCount--;
			}
			System.out.println("...");
		}
	}
}
