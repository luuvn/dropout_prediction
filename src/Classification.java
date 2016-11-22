import java.io.File;
import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/*
 Decision Tree	-> J48
 K-nearest neighbor	-> iBK
 Generalized Linear Regression -> LinearRegession
 Naive Bayes	->  Naive Bayes
 SVM	 -> LibSVM
 Neural Network -> Multilayer Perceptron (VotedPerceptron)
 */

public class Classification {

	public static void main(String[] args) throws Exception {
		File directory = new File("ARFF Converted");

		// get all the folder from a folder convert
		File[] allMajorFolder = directory.listFiles();
		for (File major : allMajorFolder) {
			if (major.isDirectory()) {
				// get all folder in major folder
				File[] allFolderInMajor = major.listFiles();
				for (File cluster : allFolderInMajor) {
					if (cluster.getName().startsWith("cluster")) {
						// get all folder in cluster folder
						File[] allFolderInCluster = cluster.listFiles();
						for (File eachYear : allFolderInCluster) {
							if (eachYear.isDirectory()) {
								new File("classifi-result/" + major.getName()
										+ "/" + eachYear.getName()).mkdirs();

								// get all file converted
								File[] allFilesConverted = eachYear.listFiles();
								for (File fileConverted : allFilesConverted) {
									if (fileConverted.getName()
											.endsWith("arff")) {

										// start process classification
										classification(fileConverted, major,
												eachYear);
									}
								}
							}
						}
					}
				}
			}
		}

	}

	public static void classification(File fileName, File major, File year)
			throws Exception {
		DataSource source = new DataSource(fileName.getPath());
		Instances data = source.getDataSet();
		// setting class attribute if the data format does not provide this
		// information
		if (data.classIndex() == -1) {
			if (year.getName().contains("Y1")) {
				data.setClassIndex(4);
			} else if (year.getName().contains("Y4Plus")) {
				data.setClassIndex(6);
			} else
				data.setClassIndex(5);
		}

		// filter
		// Remove rm = new Remove();
		// rm.setAttributeIndices("1"); // remove 1st attribute: roll number
		// remove.setInputFormat(data);
		// data = Filter.useFilter(data, rm);

		Discretize filter = new Discretize();
		filter.setOptions(Utils.splitOptions("-B 10 -M -1.0 -R first-last"));
		filter.setInputFormat(data);

		Instances filterData = Filter.useFilter(data, filter);

		int folds = 10;
		int seed = 1;
		int noi = data.numInstances();
		int newFolds = folds;
		if (noi < folds) {
			newFolds = noi;
		}
		// randomize data
		Random rand = new Random(seed);

		// use for naive bayes
		Instances filterCrossValData = new Instances(filterData);
		filterCrossValData.randomize(rand);
		if (filterCrossValData.classAttribute().isNominal())
			filterCrossValData.stratify(newFolds);
		// use for normanl numeric type data
		Instances originCrossValData = new Instances(data);
		originCrossValData.randomize(rand);
		if (originCrossValData.classAttribute().isNominal())
			originCrossValData.stratify(newFolds);

		// perform cross-validation
		Evaluation evalNaiveBayes = new Evaluation(filterCrossValData);
		Evaluation evalSVM = new Evaluation(originCrossValData);
		Evaluation evalNeuronNetwork = new Evaluation(originCrossValData);

		Evaluation evalDecisionTree = new Evaluation(originCrossValData);
		Evaluation evalIBK = new Evaluation(originCrossValData);
		Evaluation evalLR = new Evaluation(originCrossValData);

		for (int n = 0; n < newFolds; n++) {
			Instances filterTrain = filterCrossValData.trainCV(newFolds, n);
			Instances filterTest = filterCrossValData.testCV(newFolds, n);

			Instances originTrain = originCrossValData.trainCV(newFolds, n);
			Instances originTest = originCrossValData.testCV(newFolds, n);

			// build and evaluate classifier
			// naive bayes
			Classifier clsNaive = new NaiveBayes();
			clsNaive.buildClassifier(filterTrain);
			evalNaiveBayes.evaluateModel(clsNaive, filterTest);

			// SVM
			LibSVM clsSVM = new LibSVM();
			clsSVM.setOptions(Utils
					.splitOptions("-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 "
							+ "-model \"C:\\Program Files\\Weka-3-8\" -seed 1"));
			clsSVM.buildClassifier(originTrain);
			evalSVM.evaluateModel(clsSVM, originTest);

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
			vp.buildClassifier(originTrain);
			evalNeuronNetwork.evaluateModel(vp, originTest);

			// Decision Tree
			J48 j48 = new J48();
			j48.setOptions(Utils.splitOptions("-C 0.25 -M 2"));
			j48.buildClassifier(originTrain);
			evalDecisionTree.evaluateModel(j48, originTest);

			// K-nearest neighbor
			IBk ibk = new IBk();
			ibk.setOptions(Utils.splitOptions("-K 1 -W 0"));
			LinearNNSearch linearSearch = new LinearNNSearch(originTrain);
			EuclideanDistance euclideanDistance = new EuclideanDistance(
					originTest);
			euclideanDistance.setOptions(Utils.splitOptions("-R first-last"));
			linearSearch.setDistanceFunction(euclideanDistance);
			ibk.setNearestNeighbourSearchAlgorithm(linearSearch);
			ibk.buildClassifier(originTrain);
			evalIBK.evaluateModel(ibk, originTest);

			// Generalized Linear Regression
			Logistic logistic = new Logistic();
			logistic.setOptions(Utils
					.splitOptions("-R 1.0E-8 -M -1 -num-decimal-places 4"));
			logistic.buildClassifier(originTrain);
			evalLR.evaluateModel(logistic, originTest);
		}

		PrintWriter writer = new PrintWriter("classifi-result/"
				+ major.getName() + "/" + year.getName() + "/"
				+ fileName.getName() + ".txt", "UTF-8");
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
		System.out.println("Dataset: " + data.relationName());
		// System.out.println("##### Eval NaiveBayes #####");
		// System.out.println(evalNaiveBayes.toSummaryString());
		// System.out.println(evalNaiveBayes.toClassDetailsString());
		// System.out.println(evalNaiveBayes.toMatrixString());
		// System.out.println("##### Eval SVM #####");
		// System.out.println(evalSVM.toSummaryString());
		// System.out.println(evalSVM.toClassDetailsString());
		// System.out.println(evalSVM.toMatrixString());
		// System.out.println("##### Neuron Network #####");
		// System.out.println(evalNeuronNetwork.toSummaryString());
		// System.out.println(evalNeuronNetwork.toClassDetailsString());
		// System.out.println(evalNeuronNetwork.toMatrixString());
	}
}