//Classs to store trivial functions used in the classfier.
public class StatisticalHelper {
	
	
	//Compute transpose of the matrix
	public static String[][] transposeMatrix(String[][] m) {
		String[][] temp = new String[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];
		return temp;
	}

	//Compute mean of the given feature array,
	public static double computeMean(String[] featureList) {
		int length = featureList.length;
		double sum = 0.0;
		for (int i = 0; i < length; i++)
			sum = sum + Double.parseDouble(featureList[i]);
		// System.out.println("Mean : "+(sum/length));
		return sum / length;
	}

	//Compute variance of the given feature array,
	public static double computeVariance(String[] featureList, double mean) {
		double t = 0.0;
		int length = featureList.length;
		for (int i = 0; i < featureList.length; i++) {
			Double val = Double.parseDouble(featureList[i]);
			t += (val - mean) * (val - mean);
		}
		double variance = t / (length - 1);
		return variance;
	}

}
