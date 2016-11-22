import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

/**
 * 
 */

/**
 * @author luuvn
 *
 */
public class FuV2SplitData {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		CSVReader reader = new CSVReader(new FileReader("fu_v2.csv"));
		List<String[]> myEntries = reader.readAll();

		float numOfDrop = 0;
		float numOfNonDrop = 0;
		List<String[]> dropList = new ArrayList<String[]>();
		List<String[]> nonDropList = new ArrayList<String[]>();

		for (String[] i : myEntries) {
			if (i[0].contains("0")) {
				numOfNonDrop++;
				nonDropList.add(i);
			} else if (i[0].contains("1")) {
				numOfDrop++;
				dropList.add(i);
			}
		}

		long seed = System.nanoTime();
		Collections.shuffle(nonDropList, new Random(seed));
		Collections.shuffle(dropList, new Random(seed));

		int numberOfFolds = 4;
		int foldSize = (int) Math.ceil(numOfNonDrop / numberOfFolds);
		List<List<String[]>> nonDropListChunks = chunk(nonDropList, foldSize);

		foldSize = (int) Math.ceil(numOfDrop / numberOfFolds);
		List<List<String[]>> dropListChunks = chunk(dropList, foldSize);

		for (int i = 0; i < numberOfFolds; i++) {
			CSVWriter writer = new CSVWriter(new FileWriter("Data" + (i + 1)
					+ ".csv"), ',');
			nonDropListChunks.get(i).add(0, myEntries.get(0));
			writer.writeAll(nonDropListChunks.get(i));
			writer.writeAll(dropListChunks.get(i));
			writer.close();
		}
	}

	public static <T> List<List<T>> chunk(List<T> input, int chunkSize) {

		int inputSize = input.size();
		int chunkCount = (int) Math.ceil(inputSize / (double) chunkSize);

		Map<Integer, List<T>> map = new HashMap<>(chunkCount);
		List<List<T>> chunks = new ArrayList<>(chunkCount);

		for (int i = 0; i < inputSize; i++) {

			map.computeIfAbsent(i / chunkSize, (ignore) -> {

				List<T> chunk = new ArrayList<>();
				chunks.add(chunk);
				return chunk;

			}).add(input.get(i));
		}

		return chunks;
	}
}
