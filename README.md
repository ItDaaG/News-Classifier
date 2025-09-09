## AdvancedNewsClassifier

Classifies news articles from HTML files. The pipeline cleans and lemmatizes text, builds GloVe embeddings, and predicts labels with a neural network.

### Pipeline
- Practical NLP preprocessing (cleaning, stopwords, lemmatization)
- Efficient embedding lookups with hash maps
- Clear separation between data loading, preprocessing, and model steps

### Tech
Java, Maven, Stanford CoreNLP, ND4J, DeepLearning4J, JUnit

### Note
- Useful entry points: `Toolkit.java`, `ArticlesEmbedding.java`, `AdvancedNewsClassifier.java`.