import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public static Map<String, Glove> mapGloveByWord = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        mapGloveByWord = new HashMap<>();
        for (Glove glove : listGlove) {
            mapGloveByWord.put(glove.getVocabulary(), glove);
        }
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();

        for (int i = 0; i < Toolkit.vocabulary.size(); i++) {
            String word = Toolkit.vocabulary.get(i);

            boolean isStop = Toolkit.STOPWORD_SET.contains(word);

            if (!isStop) {
                Vector vector = new Vector(Toolkit.vectors.get(i));
                Glove glove = new Glove(word, vector);
                listResult.add(glove);
            }
        }

        return listResult;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        List<Integer> lengths = new ArrayList<>();

        for (ArticlesEmbedding embedding : _listEmbedding) {
            lengths.add(embedding.getDocumentLength());
        }

        lengths.sort(null);

        int size = lengths.size();

        if (size % 2 == 0) {
            int mid1 = lengths.get(size / 2);
            int mid2 = lengths.get((size / 2) + 1);
            return (mid1 + mid2) / 2;
        } else {
            return lengths.get((size + 1) / 2);
        }
    }

    public void populateEmbedding() {
        for (ArticlesEmbedding embedding: listEmbedding){
            try{
                embedding.getEmbedding();
            }
            catch (InvalidSizeException sizeException){
                embedding.setEmbeddingSize(embeddingSize);
            }
            catch (InvalidTextException textException){
                embedding.getNewsContent();
            } catch (Exception exception) {
                throw new RuntimeException(exception);
            }
        }

    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        for (ArticlesEmbedding embedding: listEmbedding){
            if(embedding.getNewsType().equals(NewsArticles.DataType.Training)){
                inputNDArray = embedding.getEmbedding();
                outputNDArray = Nd4j.zeros(1, _numberOfClasses);

                outputNDArray.putScalar(Integer.parseInt(embedding.getNewsLabel()) - 1, 1);

                DataSet myDataSet = new DataSet(inputNDArray, outputNDArray);
                listDS.add(myDataSet);
            }
        }
        return new ListDataSetIterator<>(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        for (ArticlesEmbedding article : _listEmbedding){
            if (article.getNewsType().equals(NewsArticles.DataType.Testing)){
                int[] predictions = myNeuralNetwork.predict(article.getEmbedding());

                for (int prediction : predictions){
                    listResult.add(prediction);
                    article.setNewsLabel(String.valueOf(prediction + 1));
                }
            }
        }
        return listResult;
    }

    public void printResults() {
        Map<String, List<ArticlesEmbedding>> groupedArticles = new HashMap<>();

        for (ArticlesEmbedding article : listEmbedding) {
            if (article.getNewsType() == NewsArticles.DataType.Testing) {
                groupedArticles
                    .computeIfAbsent(article.getNewsLabel(), k -> new ArrayList<>())
                    .add(article);
            }
        }

        for (String label : groupedArticles.keySet()) {
            System.out.println("Group " + label);
            for (ArticlesEmbedding article : groupedArticles.get(label)) {
                System.out.println(article.getNewsTitle());
            }
        }

    }
}