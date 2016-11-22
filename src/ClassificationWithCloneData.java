import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/**
 * Created by dgvie on 5/29/2016.
 */
public class ClassificationWithCloneData {

	// absolute path to 'ARFF Converted' folder
	static final String BASE_FOLDER = "ARFF Converted";

	public static void main(String[] args) throws Exception {
		File all_studentFolder;
		File[] filesToRead;

		File dataFolder = new File(BASE_FOLDER);
		File[] majorsFolder = dataFolder.listFiles();
		if (majorsFolder != null) {
			for (File major : majorsFolder) {

				if (major.isDirectory()) { // ignore hidden files like:
											// Desktop.ini ...
					System.out.println("\n - " + major.getPath());
					if (!major.getName().equals("major.5")) {
						all_studentFolder = new File(major.getPath()
								+ "\\all_student");
						filesToRead = all_studentFolder.listFiles();
						for (File file : filesToRead) {
							if (file.getName().endsWith(".arff")) {
								execute(file, major, file);
							}
						}
					} else {
						System.out.println(" --> skipped ");
					}
				}
			}
		}
	}

	/**
	 * Processing .arff file.
	 * 
	 * @param arffFile
	 * @throws Exception
	 */
	private static void execute(File arffFile, File major, File year)
			throws Exception {
		System.out.println("     - " + arffFile.getName());

		DataSource source = new DataSource(arffFile.getAbsolutePath());
		Instances data = source.getDataSet();

		Attribute classAttribute = data.attribute("Is_Dropout");
		data.setClass(classAttribute);

		data.randomize(new java.util.Random(0));

		int trainSize = (int) Math.round(data.numInstances() * 0.8);
		int testSize = data.numInstances() - trainSize;

		Instances trainData = new Instances(data, 0, trainSize);
		Instances testData = new Instances(data, trainSize, testSize);

		// equalizing drop & non-drop set
		trainData = balancingTrainingData(trainData, classAttribute);

		// run classification algorithms
		classify(trainData, testData, arffFile, major, year);

	}

	/**
	 * Run classification algorithms.
	 * 
	 * @param trainData
	 * @param testData
	 * @throws Exception
	 */
	private static void classify(Instances trainData, Instances testData,
			File arffFile, File major, File year) throws Exception {
		// Instances data = trainData;
		// setting class attribute if the data format does not provide this
		// information
		// if (trainData.classIndex() == -1) {
		// if (year.getName().contains("Y1")) {
		Attribute classAttribute = trainData.attribute("Is_Dropout");
		trainData.setClass(classAttribute);
		testData.setClass(classAttribute);
		// } else if (year.getName().contains("Y4Plus")) {
		// data.setClassIndex(6);
		// } else
		// data.setClassIndex(5);
		// }

		// filter
		// Remove rm = new Remove();
		// rm.setAttributeIndices("1"); // remove 1st attribute: roll number
		// remove.setInputFormat(data);
		// data = Filter.useFilter(data, rm);

		Discretize filter = new Discretize();
		filter.setOptions(Utils.splitOptions("-B 10 -M -1.0 -R first-last"));
		filter.setInputFormat(trainData);

		Instances filterData = Filter.useFilter(trainData, filter);

		Discretize filterTest = new Discretize();
		filterTest
				.setOptions(Utils.splitOptions("-B 10 -M -1.0 -R first-last"));
		filterTest.setInputFormat(testData);

		Instances TestData = Filter.useFilter(testData, filter);

		// int folds = 10;
		// int seed = 1;
		// int noi = data.numInstances();
		// int newFolds = folds;
		// if (noi < folds) {
		// newFolds = noi;
		// }
		// randomize data

		// use for naive bayes
		Instances filterTrainData = new Instances(filterData);
		Instances filterTestData = new Instances(TestData);
		// use for normanl numeric type data
		Instances originTrainData = new Instances(trainData);

		// perform cross-validation
		Evaluation evalNaiveBayes = new Evaluation(filterTrainData);
		Evaluation evalSVM = new Evaluation(originTrainData);
		Evaluation evalNeuronNetwork = new Evaluation(originTrainData);

		Evaluation evalDecisionTree = new Evaluation(originTrainData);
		Evaluation evalIBK = new Evaluation(originTrainData);
		Evaluation evalLR = new Evaluation(originTrainData);

		// build and evaluate classifier
		// naive bayes
		Classifier clsNaive = new NaiveBayes();
		clsNaive.buildClassifier(filterTrainData);
		evalNaiveBayes.evaluateModel(clsNaive, filterTestData);

		// SVM
		LibSVM clsSVM = new LibSVM();
		clsSVM.setOptions(Utils
				.splitOptions("-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 "
						+ "-model \"C:\\Program Files\\Weka-3-8\" -seed 1"));
		clsSVM.buildClassifier(originTrainData);
		evalSVM.evaluateModel(clsSVM, testData);

		// Neuron Network
		// MultilayerPerceptron mlp = new MultilayerPerceptron();
		// // L = Learning Rate
		// // M = Momentum
		// // N = Training Time or Epochs
		// // H = Hidden Layers //
		// mlp.setOptions(Utils
		// .splitOptions("-L 0.1 -M 0.2 -N 2000 -V 0 -S 0 -E 20 -H 3"));
		// mlp.buildClassifier(originTrain);
		// evalNeuronNetwork.evaluateModel(mlp, originTest);

		// Neuron Network
		VotedPerceptron vp = new VotedPerceptron();
		vp.setOptions(Utils.splitOptions("-I 1 -E 1.0 -S 1 -M 10000"));
		vp.buildClassifier(originTrainData);
		evalNeuronNetwork.evaluateModel(vp, testData);

		// Decision Tree
		J48 j48 = new J48();
		j48.setOptions(Utils.splitOptions("-C 0.25 -M 2"));
		j48.buildClassifier(originTrainData);
		evalDecisionTree.evaluateModel(j48, testData);

		// K-nearest neighbor
		IBk ibk = new IBk();
		ibk.setOptions(Utils.splitOptions("-K 1 -W 0"));
		LinearNNSearch linearSearch = new LinearNNSearch(originTrainData);
		EuclideanDistance euclideanDistance = new EuclideanDistance(testData);
		euclideanDistance.setOptions(Utils.splitOptions("-R first-last"));
		linearSearch.setDistanceFunction(euclideanDistance);
		ibk.setNearestNeighbourSearchAlgorithm(linearSearch);
		ibk.buildClassifier(originTrainData);
		evalIBK.evaluateModel(ibk, testData);

		// Generalized Linear Regression
		Logistic logistic = new Logistic();
		logistic.setOptions(Utils
				.splitOptions("-R 1.0E-8 -M -1 -num-decimal-places 4"));
		logistic.buildClassifier(originTrainData);
		evalLR.evaluateModel(logistic, testData);

		new File("classifi-result/" + major.getName() + "/" + year.getName())
				.mkdirs();
		PrintWriter writer = new PrintWriter("classifi-result/"
				+ major.getName() + "/" + year.getName() + "/"
				+ arffFile.getName() + ".txt", "UTF-8");
		// writer.println("Dataset: " + data.relationName());
		writer.println("##### NaiveBayes #####");
		writer.println(evalNaiveBayes.toSummaryString());
		writer.println(evalNaiveBayes.toClassDetailsString());
		writer.println(evalNaiveBayes.toMatrixString());
		writer.println("##### SVM #####");
		writer.println(evalSVM.toSummaryString());
		writer.println(evalSVM.toClassDetailsString());
		writer.println(evalSVM.toMatrixString());
		writer.println("##### Neuron Network #####");
		writer.println(evalNeuronNetwork.toSummaryString());
		writer.println(evalNeuronNetwork.toClassDetailsString());
		writer.println(evalNeuronNetwork.toMatrixString());
		writer.println("##### Decision Tree #####");
		writer.println(evalDecisionTree.toSummaryString());
		writer.println(evalDecisionTree.toClassDetailsString());
		writer.println(evalDecisionTree.toMatrixString());
		writer.println("##### K-nearest neighbor #####");
		writer.println(evalIBK.toSummaryString());
		writer.println(evalIBK.toClassDetailsString());
		writer.println(evalIBK.toMatrixString());
		writer.println("##### Logistic Regression #####");
		writer.println(evalLR.toSummaryString());
		writer.println(evalLR.toClassDetailsString());
		writer.println(evalLR.toMatrixString());
		writer.close();

		// output evaluation
		// System.out.println();
		System.out.println("##### Setup #####");
		System.out.println("Dataset: " + trainData.relationName());
	}

	/**
	 * balancing the number of drop out & non-dropout in training set.
	 * 
	 * @param trainData
	 * @param classAttribute
	 * @return
	 */
	private static Instances balancingTrainingData(Instances trainData,
			Attribute classAttribute) {
		int numOfDrop = 0;
		int numOfNonDrop = 0;
		List<Instance> dropList = new ArrayList<Instance>();
		List<Instance> nonDropList = new ArrayList<Instance>();

		for (Instance i : trainData) {
			if (i.value(classAttribute) == 0.0) {
				numOfNonDrop++;
				nonDropList.add(i);
			} else {
				dropList.add(i);
				numOfDrop++;
			}
		}

		int difference = numOfNonDrop - numOfDrop;

		while (difference > 0 && dropList.size() > 0) {
			for (Instance dropInstance : dropList) {
				if (difference <= 0) {
					break;
				}
				difference--;
				trainData.add(dropInstance);
			}
		}
		while (difference < 0 && nonDropList.size() > 0) {
			for (Instance nonDropInstance : nonDropList) {
				if (difference >= 0) {
					break;
				}
				difference++;
				trainData.add(nonDropInstance);
			}
		}
		return trainData;
	}
}