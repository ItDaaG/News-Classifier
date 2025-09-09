import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> vocabulary = null;
    public static List<double[]> vectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};
    public static final Set<String> STOPWORD_SET = new HashSet<>(Arrays.asList(STOPWORDS));

    public void loadGlove() throws IOException {
        File gloveFile = getFileFromResource(FILENAME_GLOVE);

        vocabulary = new ArrayList<>();
        vectors = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(gloveFile.toPath())) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");

                if (tokens.length >= 2) {
                    vocabulary.add(tokens[0]);

                    double[] vector = new double[tokens.length - 1];
                    for (int i = 1; i < tokens.length; i++) {
                        vector[i - 1] = Double.parseDouble(tokens[i]);
                    }
                    vectors.add(vector);
                }
            }
        }
        catch (IOException e) {
            System.out.println("Error loading Glove file: " + e.getMessage());
            throw e;
        }
    }



    private static File getFileFromResource(String fileName) throws IOException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.getPath());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> newsArticles = new ArrayList<>();

        try (Stream<Path> paths = Files.walk(Paths.get("src/main/resources/News"))) {
            paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".htm"))
                    .forEach(path -> {
                        try {
                            String htmlCode = Files.readString(path);

                            String newsTitle = HtmlParser.getNewsTitle(htmlCode);
                            String newsContent = HtmlParser.getNewsContent(htmlCode);
                            NewsArticles.DataType dataType = HtmlParser.getDataType(htmlCode);
                            String label = HtmlParser.getLabel(htmlCode);

                            NewsArticles newsArticle = new NewsArticles(newsTitle, newsContent, dataType, label);
                            newsArticles.add(newsArticle);
                        } catch (IOException e) {
                            System.out.println("Error reading file: " + path.toString() + " - " + e.getMessage());
                        }
                    });
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return newsArticles;
    }

    public static List<String> getListVocabulary() {
        return vocabulary;
    }

    public static List<double[]> getlistVectors() {
        return vectors;
    }
}
