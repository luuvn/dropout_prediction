import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * @author luuvn
 *
 */
public class FuV2Classification {
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		int numOfData = 4;
		Instances[] allInstance = new Instances[4];

		for (int i = 0; i < numOfData; i++) {
			Instances getInstance = getInstance("Data" + (i + 1) + ".csv");
			// normalization
			Instances processedInstance = normalization(getInstance);

			allInstance[i] = processedInstance;
		}

		// Use a set of classifiers
		Classifier clsNaive = new NaiveBayes();

		// SVM
		LibSVM clsSVM = new LibSVM();
		// clsSVM.setOptions(Utils.splitOptions("-c 100 -e 0.125"));

		// Neuron Network
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		// L = Learning Rate
		// M = Momentum
		// N = Training Time or Epochs
		// H = Hidden Layers //
		// mlp.setOptions(Utils
		// .splitOptions("-L 0.1 -M 0.2 -N 2000 -V 0 -S 0 -E 20 -H 3"));

		// Decision Tree
		J48 j48 = new J48();
		// j48.setOptions(Utils.splitOptions("-C 0.25 -M 2"));

		// K-nearest neighbor
		IBk ibk = new IBk();
		// ibk.setOptions(Utils.splitOptions("-K 1 -W 0"));
		LinearNNSearch linearSearch = new LinearNNSearch(allInstance[1]);
		EuclideanDistance euclideanDistance = new EuclideanDistance(
				allInstance[0]);
		euclideanDistance.setOptions(Utils.splitOptions("-R first-last"));
		linearSearch.setDistanceFunction(euclideanDistance);
		ibk.setNearestNeighbourSearchAlgorithm(linearSearch);

		// Generalized Linear Regression
		Logistic logistic = new Logistic();
		// logistic.setOptions(Utils
		// .splitOptions("-R 1.0E-8 -M -1 -num-decimal-places 4"));

		Classifier[] models = { clsNaive, clsSVM, mlp, j48, ibk, logistic };
		// Classifier[] models = { clsSVM };

		// Prepare data to build model
		int classIndex = 0;

		// Get result
		for (int i = 0; i < allInstance.length; i++) {
			int testSetId = i;
			int normalSetId = (i + 1) >= allInstance.length ? i + 1
					- allInstance.length : i + 1;
			int cloneSetId = (i + 2) >= allInstance.length ? i + 2
					- allInstance.length : i + 2;
			int clusterSetId = (i + 3) >= allInstance.length ? i + 3
					- allInstance.length : i + 3;
			// test set
			Instances testSet = cloneInstance(allInstance[testSetId]);
			testSet.setClassIndex(classIndex);
			// normal training set
			Instances normalTraining = cloneInstance(allInstance[normalSetId]);
			normalTraining.setClassIndex(classIndex);
			// clone training set
			Instances cloneTraining = balancingTrainingData(cloneInstance(allInstance[cloneSetId]));
			cloneTraining.setClassIndex(classIndex);
			// cluster training set
			Instances clusterTraining = cloneInstance(allInstance[clusterSetId]);
			Instances[] allClusterBuild = buildSameSizeKMean(clusterTraining);

			// Get result
			writeResult("" + i + "/NormalTraining", models, normalTraining,
					testSet);
			writeResult("" + i + "/CloneTraining", models, cloneTraining,
					testSet);
			for (int j = 0; j < allClusterBuild.length; j++) {
				allClusterBuild[j].setClassIndex(classIndex);
				writeResult("" + i + "/ClusterTraining_" + j, models,
						allClusterBuild[j], testSet);
			}
		}
	}

	/**
	 * 
	 * @param fileName
	 * @param models
	 * @param training
	 * @param test
	 * @throws Exception
	 */
	public static void writeResult(String fileName, Classifier[] models,
			Instances training, Instances test) throws Exception {
		PrintWriter writer = new PrintWriter("classifi-result/FuV2/" + fileName
				+ ".txt", "UTF-8");

		for (int i = 0; i < models.length; i++) {
			writer.println("##############################################");
			writer.println(models[i].getClass().getCanonicalName());
			writer.println("##############################################");

			Evaluation validation = classify(models[i], training, test);
			writer.println(validation.toSummaryString());
			writer.println(validation.toClassDetailsString());
			writer.println(validation.toMatrixString());
		}
		writer.close();
	}

	/**
	 * 
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public static Instances normalization(Instances instance) throws Exception {
		Normalize norm = new Normalize();
		norm.setInputFormat(instance);

		Instances processedInstance = Filter.useFilter(instance, norm);

		return processedInstance;
	}

	/**
	 * 
	 * @param fileName
	 * @return
	 * @throws Exception
	 */
	public static Instances getInstance(String fileName) throws Exception {
		File file = new File(fileName);
		DataSource source = new DataSource(file.getPath());
		Instances instance = source.getDataSet();
		// instance.setClassIndex(0);

		// Convert class to nominal
		NumericToNominal convert = new NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "first"; // range of variables to make numeric

		convert.setOptions(options);
		convert.setInputFormat(instance);
		Instances newData = Filter.useFilter(instance, convert);

		System.out.println("Instances length: " + newData.numInstances());
		return newData;
	}

	/**
	 * 
	 * @param model
	 * @param trainingSet
	 * @param testingSet
	 * @return
	 * @throws Exception
	 */
	public static Evaluation classify(Classifier model, Instances trainingSet,
			Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);

		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);

		System.out.println("##############################################");
		System.out.println(model.getClass().getCanonicalName());
		System.out.println("##############################################");
		System.out.println(evaluation.toSummaryString());
		System.out.println(evaluation.toClassDetailsString());
		System.out.println(evaluation.toMatrixString());

		return evaluation;
	}

	/**
	 * balancing the number of drop out & non-dropout in training set.
	 * 
	 * @param trainData
	 * @param classAttribute
	 * @return
	 */
	private static Instances balancingTrainingData(Instances trainData) {
		Instances newIns = cloneInstance(trainData);

		Attribute classAttribute = trainData.attribute(0);

		List<Instance> dropList = new ArrayList<Instance>();
		List<Instance> nonDropList = new ArrayList<Instance>();

		for (Instance i : trainData) {
			if (i.value(classAttribute) == 0) {
				nonDropList.add(i);
			} else {
				dropList.add(i);
			}
		}

		long seed = System.nanoTime();
		Collections.shuffle(nonDropList, new Random(seed));
		Collections.shuffle(dropList, new Random(seed));

		int difference = nonDropList.size() - dropList.size();

		while (difference > 0 && dropList.size() > 0) {
			for (Instance dropInstance : dropList) {
				if (difference <= 0) {
					break;
				}
				difference--;
				newIns.add(dropInstance);
			}
		}
		while (difference < 0 && nonDropList.size() > 0) {
			for (Instance nonDropInstance : nonDropList) {
				if (difference >= 0) {
					break;
				}
				difference++;
				newIns.add(nonDropInstance);
			}
		}
		return newIns;
	}

	/**
	 * 
	 * @param trainingData
	 * @return
	 * @throws Exception
	 */
	public static Instances[] buildSameSizeKMean(Instances trainingData)
			throws Exception {
		Instances dropIns = new Instances(trainingData, 0);
		Instances nonDropIns = new Instances(trainingData, 0);

		Attribute classAttribute = trainingData.attribute(0);
		for (Instance i : trainingData) {
			if (i.value(classAttribute) == 0) {
				nonDropIns.add(i);
			} else {
				dropIns.add(i);
			}
		}

		int expectedNumberOFClusters = (int) Math.ceil((double) nonDropIns
				.size() / dropIns.size());

		Instances[] allClusterNonDrop = executeSameSizeKMeans(nonDropIns,
				expectedNumberOFClusters);
		Instances[] allCombinationBuild = new Instances[allClusterNonDrop.length];

		for (int i = 0; i < allClusterNonDrop.length; i++) {
			allCombinationBuild[i] = new Instances(trainingData, 0);
			// copy all in ClusterNonDrop
			for (Instance cluster : allClusterNonDrop[i]) {
				allCombinationBuild[i].add(cluster);
			}
			// copy all in dropIns
			for (Instance drop : dropIns) {
				allCombinationBuild[i].add(drop);
			}
		}

		return allCombinationBuild;
	}

	/**
	 * 
	 * @param nonDropIns
	 * @param expectedNumberOFClusters
	 * @return
	 * @throws Exception
	 */
	private static Instances[] executeSameSizeKMeans(Instances nonDropIns,
			int expectedNumberOFClusters) throws Exception {
		SimpleKMeans kmeans = new SimpleKMeans();

		int seed = (int) System.nanoTime();
		kmeans.setSeed(seed);

		// important parameter to set: preserver order, number of cluster.
		kmeans.setPreserveInstancesOrder(true);
		kmeans.setNumClusters(expectedNumberOFClusters);

		int numInstances = nonDropIns.numInstances();
		int expectedClusterSize = (int) Math.ceil((double) numInstances
				/ expectedNumberOFClusters);

		// create the model
		kmeans.buildClusterer(nonDropIns);

		// print out the cluster centroids
		Instances centroids = kmeans.getClusterCentroids();

		EuclideanDistance dist = (EuclideanDistance) kmeans
				.getDistanceFunction();

		Map<Integer, List<Instance>> map = new HashMap<Integer, List<Instance>>();

		// get cluster membership for each instance
		for (int i = 0; i < nonDropIns.numInstances(); i++) {

			if (map.get(kmeans.clusterInstance(nonDropIns.instance(i))) == null) {
				map.put(kmeans.clusterInstance(nonDropIns.instance(i)),
						new LinkedList<Instance>(Arrays.asList(nonDropIns
								.instance(i))));
			} else {
				map.get(kmeans.clusterInstance(nonDropIns.instance(i))).add(
						nonDropIns.instance(i));
			}
		}

		int largestClusterPos;
		List<Integer> ignoreItems = new ArrayList<Integer>();
		for (int i = 0; i < kmeans.getNumClusters() - 1; i++) {
			largestClusterPos = SameSizeKmeansClustering
					.getPositionOfLargestCluster(map, ignoreItems);
			ignoreItems.add(largestClusterPos);

			SameSizeKmeansClustering.quickSort(map.get(largestClusterPos),
					centroids.instance(largestClusterPos), dist, 0,
					map.get(largestClusterPos).size() - 1);

			int currentSize = map.get(largestClusterPos).size();

			for (int j = currentSize - 1; j >= expectedClusterSize; j--) {
				SameSizeKmeansClustering.moveToOtherCLuster(
						map.get(largestClusterPos).remove(j), map, centroids,
						dist, ignoreItems);
			}
		}

		Instances[] allClusterInstances = new Instances[expectedNumberOFClusters];
		// System.out.println("expectedNumberOFClusters: ");
		for (int i = 0; i < map.size(); i++) {
			allClusterInstances[i] = new Instances(nonDropIns, 0);
			for (Instance ins : map.get(i)) {
				allClusterInstances[i].add(ins);
			}
			// System.out.println("allClusterInstances[" + i + "] size: "
			// + allClusterInstances[i].size());
		}

		return allClusterInstances;
	}

	public static Instances cloneInstance(Instances ins) {
		Instances newIns = new Instances(ins, 0);
		for (Instance i : ins) {
			newIns.add(i);
		}
		return newIns;
	}
}
