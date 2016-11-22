import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Viet Trung
 */
public class SameSizeKmeansClustering {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	static final String BASE_FOLDER = "\\data\\ARFF Converted";

	public static void main(String[] args) throws Exception {

		File nonDrop;
		File[] filesToRead;

		File dataFolder = new File(BASE_FOLDER);
		File[] majorsFolder = dataFolder.listFiles();
		if (majorsFolder != null) {
			for (File major : majorsFolder) {
				if (!major.getName().equals("desktop.ini")) {
					System.out.println("\n - " + major.getPath());
					if (!major.getName().equals("major.5")) {
						nonDrop = new File(major.getPath()
								+ "\\non-drop_student");
						filesToRead = nonDrop.listFiles();
						for (File file : filesToRead) {
							if (file.getName().endsWith(".arff")) {
								execute(file);
							}
						}
					} else {
						System.out.println(" --> skipped ");
					}
				}
			}
		}

	}

	private static String getFileNameWithoutExtension(File file) {
		String fileName = file.getName();
		int pos = fileName.lastIndexOf(".");
		if (pos > 0) {
			fileName = fileName.substring(0, pos);
		}
		return fileName;
	}

	private static void execute(File arffFile) throws Exception {
		System.out.println("     - " + arffFile.getName());

		try {
			int expectedNumberOFClusters = getExpectedNumberOFClusters(arffFile
					.getParent()
					+ "\\"
					+ getFileNameWithoutExtension(arffFile)
					+ ".csv");
			executeSameSizeKMeans(arffFile, expectedNumberOFClusters);
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}

	private static void executeSameSizeKMeans(File arffFile,
			int expectedNumberOFClusters) throws Exception {
		BufferedReader datafile = readDataFile(arffFile.getAbsolutePath());
		Instances data = new Instances(datafile);

		// long start = System.currentTimeMillis();

		SimpleKMeans kmeans = new SimpleKMeans();

		kmeans.setSeed(10);

		// important parameter to set: preserver order, number of cluster.
		kmeans.setPreserveInstancesOrder(true);
		kmeans.setNumClusters(expectedNumberOFClusters);

		int numInstances = data.numInstances();
		int expectedClusterSize = (int) Math.ceil((double) numInstances
				/ kmeans.getNumClusters());

		// create the model
		kmeans.buildClusterer(data);

		// print out the cluster centroids
		Instances centroids = kmeans.getClusterCentroids();

		EuclideanDistance dist = (EuclideanDistance) kmeans
				.getDistanceFunction();

		Map<Integer, List<Instance>> map = new HashMap<Integer, List<Instance>>();

		// get cluster membership for each instance
		for (int i = 0; i < data.numInstances(); i++) {

			if (map.get(kmeans.clusterInstance(data.instance(i))) == null) {
				map.put(kmeans.clusterInstance(data.instance(i)),
						new LinkedList<Instance>(
								Arrays.asList(data.instance(i))));
			} else {
				map.get(kmeans.clusterInstance(data.instance(i))).add(
						data.instance(i));
			}
		}

		int largestClusterPos;
		List<Integer> ignoreItems = new ArrayList<Integer>();
		for (int i = 0; i < kmeans.getNumClusters() - 1; i++) {
			largestClusterPos = getPositionOfLargestCluster(map, ignoreItems);
			ignoreItems.add(largestClusterPos);

			quickSort(map.get(largestClusterPos),
					centroids.instance(largestClusterPos), dist, 0,
					map.get(largestClusterPos).size() - 1);

			int currentSize = map.get(largestClusterPos).size();

			for (int j = currentSize - 1; j >= expectedClusterSize; j--) {
				moveToOtherCLuster(map.get(largestClusterPos).remove(j), map,
						centroids, dist, ignoreItems);
			}
		}

		System.out.print("       " + expectedNumberOFClusters + ": ");
		List<Instance> instances;
		for (int i = 0; i < map.size(); i++) {
			System.out.print(map.get(i).size() + "; ");
			writeToArff(arffFile, map.get(i), i);
		}
		System.out.println();
		// System.out.println("       Elapsed time: " +
		// (System.currentTimeMillis()-start));
	}

	private static int getExpectedNumberOFClusters(String filePath)
			throws Exception {

		String content = new String(Files.readAllBytes(Paths.get(filePath)));
		int result = (int) Double.parseDouble(content);

		return result > 0 ? result : 1;
	}

	private static void writeToArff(File originalFile,
			List<Instance> instances, int clusterNo) {
		try {

			Path path = Paths.get(originalFile.getParent() + "\\clusters\\"
					+ getFileNameWithoutExtension(originalFile));
			Files.createDirectories(path);

			List<String> lines = new ArrayList<String>();
			boolean firstIns = true;
			lines.add("@relation student\n");
			for (Instance ins : instances) {
				if (firstIns) {
					Enumeration test = ins.enumerateAttributes();
					while (test.hasMoreElements()) {
						lines.add(test.nextElement().toString());
					}
					lines.add("\n@data");

					firstIns = false;
				}

				lines.add(ins.toString());
			}

			Path file = Paths.get(originalFile.getParent() + "\\clusters\\"
					+ getFileNameWithoutExtension(originalFile) + "\\cluster"
					+ clusterNo + ".arff");
			Files.write(file, lines, Charset.forName("UTF-8"));

		} catch (IOException e) {
			e.printStackTrace();
		}

		// for (Instance ins : instances) {
		// Enumeration test = ins.enumerateAttributes();
		// while (test.hasMoreElements()) {
		// System.out.println(test.nextElement().toString());
		// }

		// }
	}

	static void moveToOtherCLuster(Instance instanceToBeMoved,
			Map<Integer, List<Instance>> map, Instances centroids,
			EuclideanDistance dist, List<Integer> ignoreItems) {

		double longestDistance = 0;
		int pos2Move = 0;
		for (int i = 0; i < centroids.numInstances(); i++) {
			if (!ignoreItems.contains(i)) {
				if (dist.distance(centroids.instance(i), instanceToBeMoved) > longestDistance) {
					longestDistance = dist.distance(centroids.instance(i),
							instanceToBeMoved);
					pos2Move = i;
				}
			}
		}
		map.get(pos2Move).add(instanceToBeMoved);
	}

	static int getPositionOfLargestCluster(
			Map<Integer, List<Instance>> map, List<Integer> ignoreItems) {
		int result = 0;
		int currentLargestSize = 0;
		for (int i = 0; i < map.size(); i++) {
			if (!ignoreItems.contains(i)
					&& map.get(i).size() > currentLargestSize) {
				currentLargestSize = map.get(i).size();
				result = i;
			}
		}
		return result;
	}

	/*
	 * In bubble sort, we basically traverse the array from first to
	 * array_length - 1 position and compare the element with the next one.
	 * Element is swapped with the next element if the next element is greater.
	 * 
	 * Bubble sort steps are as follows.
	 * 
	 * 1. Compare array[0] & array[1] 2. If array[0] > array [1] swap it. 3.
	 * Compare array[1] & array[2] 4. If array[1] > array[2] swap it. ... 5.
	 * Compare array[n-1] & array[n] 6. if [n-1] > array[n] then swap it.
	 * 
	 * After this step we will have largest element at the last index.
	 * 
	 * Repeat the same steps for array[1] to array[n-1]
	 */
	private static void bubbleSort(List<Instance> instances, Instance centroid,
			EuclideanDistance dist) {
		int n = instances.size();
		Instance temp;

		for (int i = 0; i < n; i++) {
			for (int j = 1; j < (n - i); j++) {

				if (dist.distance(centroid, instances.get(j - 1)) > dist
						.distance(centroid, instances.get(j))) {
					// swap the elements!
					temp = instances.get(j - 1);
					instances.set(j - 1, instances.get(j));
					instances.set(j, temp);
				}

			}
		}

	}

	public static void quickSort(List<Instance> instances, Instance centroid,
			EuclideanDistance dist, int low, int high) {
		if (instances == null || instances.size() == 0)
			return;

		if (low >= high) {
			return;
		}

		// pick the pivot
		int middle = low + (high - low) / 2;
		double pivot = dist.distance(centroid, instances.get(middle));

		// make left < pivot and right > pivot
		int i = low, j = high;
		while (i <= j) {
			while (dist.distance(centroid, instances.get(i)) < pivot) {
				i++;
			}

			while (dist.distance(centroid, instances.get(j)) > pivot) {
				j--;
			}

			if (i <= j) {
				Instance temp = instances.get(i);
				instances.set(i, instances.get(j));
				instances.set(j, temp);
				i++;
				j--;
			}
		}

		// recursively sort two sub parts
		if (low < j)
			quickSort(instances, centroid, dist, low, j);

		if (high > i)
			quickSort(instances, centroid, dist, i, high);
	}
}