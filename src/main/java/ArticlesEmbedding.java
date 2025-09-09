import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Properties;
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;



public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";
    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        super(_title, _content, _type, _label);
    }

    public void setEmbeddingSize(int _size) {
        intSize = _size;
    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        if (processedText.isEmpty()) {

            String originalContent = super.getNewsContent();
            String cleanContent = textCleaning(originalContent);

            Properties props = new Properties();
            props.setProperty("annotators", "tokenize,pos,lemma");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
            CoreDocument document = pipeline.processToCoreDocument(cleanContent);

            StringBuilder lemmatizedTextBuilder = new StringBuilder();

            for (CoreLabel token : document.tokens()) {
                lemmatizedTextBuilder.append(token.lemma()).append(" ");
            }

            String lemmatizedText = lemmatizedTextBuilder.toString();

            Set<String> stopwords = new HashSet<>(Arrays.asList(Toolkit.STOPWORDS));
            StringBuilder filtered = new StringBuilder();
            for (String word : lemmatizedText.split("\\s+")) {
                if (!stopwords.contains(word)) {
                    filtered.append(word).append(" ");
                }
            }
            lemmatizedText = filtered.toString().trim();

            processedText = lemmatizedText.toLowerCase();
        }
        return processedText;
        }

        public int getDocumentLength() {
            if (processedText.isEmpty()) {
                getNewsContent();
            }
            int count = 0;
            for (String word : processedText.split("\\s+")) {
                if (AdvancedNewsClassifier.mapGloveByWord.containsKey(word)) {
                    count++;
                }
            }
            return count;
        }


    public INDArray getEmbedding() throws Exception {
        if (!newsEmbedding.isEmpty()){
            return newsEmbedding;
        }
        if (intSize == -1) {
            throw new InvalidSizeException("Invalid size");
        }

        if (processedText.isEmpty()) {
            throw new InvalidTextException("Invalid text");
        }

        int vectorSize = AdvancedNewsClassifier.listGlove.get(0).getVector().getVectorSize();
        newsEmbedding = Nd4j.zeros(intSize, vectorSize);
        String[] words = processedText.split("\\s+");
        int rowcount = 0;

        for (String word : words) {
            if (rowcount >= intSize) {
                break;
            }

            Glove glove = AdvancedNewsClassifier.mapGloveByWord.get(word);
            if (glove != null) {
                newsEmbedding.putRow(rowcount, Nd4j.create(glove.getVector().getAllElements()));
                rowcount++;
            }
        }
        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
